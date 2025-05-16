# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
    LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.deprecation import deprecate_kwarg
from .configuration_llama import LlamaConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "meta-llama/Llama-2-7b-hf"
_CONFIG_FOR_DOC = "LlamaConfig"
change_me_layers_num = 16


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]

        hidden_shape = (*input_shape, -1, self.head_dim)
        # perm_q = [0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39, 8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47, 16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55, 24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63, 64, 96, 65, 97, 66, 98, 67, 99, 68, 100, 69, 101, 70, 102, 71, 103, 72, 104, 73, 105, 74, 106, 75, 107, 76, 108, 77, 109, 78, 110, 79, 111, 80, 112, 81, 113, 82, 114, 83, 115, 84, 116, 85, 117, 86, 118, 87, 119, 88, 120, 89, 121, 90, 122, 91, 123, 92, 124, 93, 125, 94, 126, 95, 127, 128, 160, 129, 161, 130, 162, 131, 163, 132, 164, 133, 165, 134, 166, 135, 167, 136, 168, 137, 169, 138, 170, 139, 171, 140, 172, 141, 173, 142, 174, 143, 175, 144, 176, 145, 177, 146, 178, 147, 179, 148, 180, 149, 181, 150, 182, 151, 183, 152, 184, 153, 185, 154, 186, 155, 187, 156, 188, 157, 189, 158, 190, 159, 191, 192, 224, 193, 225, 194, 226, 195, 227, 196, 228, 197, 229, 198, 230, 199, 231, 200, 232, 201, 233, 202, 234, 203, 235, 204, 236, 205, 237, 206, 238, 207, 239, 208, 240, 209, 241, 210, 242, 211, 243, 212, 244, 213, 245, 214, 246, 215, 247, 216, 248, 217, 249, 218, 250, 219, 251, 220, 252, 221, 253, 222, 254, 223, 255, 256, 288, 257, 289, 258, 290, 259, 291, 260, 292, 261, 293, 262, 294, 263, 295, 264, 296, 265, 297, 266, 298, 267, 299, 268, 300, 269, 301, 270, 302, 271, 303, 272, 304, 273, 305, 274, 306, 275, 307, 276, 308, 277, 309, 278, 310, 279, 311, 280, 312, 281, 313, 282, 314, 283, 315, 284, 316, 285, 317, 286, 318, 287, 319, 320, 352, 321, 353, 322, 354, 323, 355, 324, 356, 325, 357, 326, 358, 327, 359, 328, 360, 329, 361, 330, 362, 331, 363, 332, 364, 333, 365, 334, 366, 335, 367, 336, 368, 337, 369, 338, 370, 339, 371, 340, 372, 341, 373, 342, 374, 343, 375, 344, 376, 345, 377, 346, 378, 347, 379, 348, 380, 349, 381, 350, 382, 351, 383, 384, 416, 385, 417, 386, 418, 387, 419, 388, 420, 389, 421, 390, 422, 391, 423, 392, 424, 393, 425, 394, 426, 395, 427, 396, 428, 397, 429, 398, 430, 399, 431, 400, 432, 401, 433, 402, 434, 403, 435, 404, 436, 405, 437, 406, 438, 407, 439, 408, 440, 409, 441, 410, 442, 411, 443, 412, 444, 413, 445, 414, 446, 415, 447, 448, 480, 449, 481, 450, 482, 451, 483, 452, 484, 453, 485, 454, 486, 455, 487, 456, 488, 457, 489, 458, 490, 459, 491, 460, 492, 461, 493, 462, 494, 463, 495, 464, 496, 465, 497, 466, 498, 467, 499, 468, 500, 469, 501, 470, 502, 471, 503, 472, 504, 473, 505, 474, 506, 475, 507, 476, 508, 477, 509, 478, 510, 479, 511, 512, 544, 513, 545, 514, 546, 515, 547, 516, 548, 517, 549, 518, 550, 519, 551, 520, 552, 521, 553, 522, 554, 523, 555, 524, 556, 525, 557, 526, 558, 527, 559, 528, 560, 529, 561, 530, 562, 531, 563, 532, 564, 533, 565, 534, 566, 535, 567, 536, 568, 537, 569, 538, 570, 539, 571, 540, 572, 541, 573, 542, 574, 543, 575, 576, 608, 577, 609, 578, 610, 579, 611, 580, 612, 581, 613, 582, 614, 583, 615, 584, 616, 585, 617, 586, 618, 587, 619, 588, 620, 589, 621, 590, 622, 591, 623, 592, 624, 593, 625, 594, 626, 595, 627, 596, 628, 597, 629, 598, 630, 599, 631, 600, 632, 601, 633, 602, 634, 603, 635, 604, 636, 605, 637, 606, 638, 607, 639, 640, 672, 641, 673, 642, 674, 643, 675, 644, 676, 645, 677, 646, 678, 647, 679, 648, 680, 649, 681, 650, 682, 651, 683, 652, 684, 653, 685, 654, 686, 655, 687, 656, 688, 657, 689, 658, 690, 659, 691, 660, 692, 661, 693, 662, 694, 663, 695, 664, 696, 665, 697, 666, 698, 667, 699, 668, 700, 669, 701, 670, 702, 671, 703, 704, 736, 705, 737, 706, 738, 707, 739, 708, 740, 709, 741, 710, 742, 711, 743, 712, 744, 713, 745, 714, 746, 715, 747, 716, 748, 717, 749, 718, 750, 719, 751, 720, 752, 721, 753, 722, 754, 723, 755, 724, 756, 725, 757, 726, 758, 727, 759, 728, 760, 729, 761, 730, 762, 731, 763, 732, 764, 733, 765, 734, 766, 735, 767, 768, 800, 769, 801, 770, 802, 771, 803, 772, 804, 773, 805, 774, 806, 775, 807, 776, 808, 777, 809, 778, 810, 779, 811, 780, 812, 781, 813, 782, 814, 783, 815, 784, 816, 785, 817, 786, 818, 787, 819, 788, 820, 789, 821, 790, 822, 791, 823, 792, 824, 793, 825, 794, 826, 795, 827, 796, 828, 797, 829, 798, 830, 799, 831, 832, 864, 833, 865, 834, 866, 835, 867, 836, 868, 837, 869, 838, 870, 839, 871, 840, 872, 841, 873, 842, 874, 843, 875, 844, 876, 845, 877, 846, 878, 847, 879, 848, 880, 849, 881, 850, 882, 851, 883, 852, 884, 853, 885, 854, 886, 855, 887, 856, 888, 857, 889, 858, 890, 859, 891, 860, 892, 861, 893, 862, 894, 863, 895, 896, 928, 897, 929, 898, 930, 899, 931, 900, 932, 901, 933, 902, 934, 903, 935, 904, 936, 905, 937, 906, 938, 907, 939, 908, 940, 909, 941, 910, 942, 911, 943, 912, 944, 913, 945, 914, 946, 915, 947, 916, 948, 917, 949, 918, 950, 919, 951, 920, 952, 921, 953, 922, 954, 923, 955, 924, 956, 925, 957, 926, 958, 927, 959, 960, 992, 961, 993, 962, 994, 963, 995, 964, 996, 965, 997, 966, 998, 967, 999, 968, 1000, 969, 1001, 970, 1002, 971, 1003, 972, 1004, 973, 1005, 974, 1006, 975, 1007, 976, 1008, 977, 1009, 978, 1010, 979, 1011, 980, 1012, 981, 1013, 982, 1014, 983, 1015, 984, 1016, 985, 1017, 986, 1018, 987, 1019, 988, 1020, 989, 1021, 990, 1022, 991, 1023, 1024, 1056, 1025, 1057, 1026, 1058, 1027, 1059, 1028, 1060, 1029, 1061, 1030, 1062, 1031, 1063, 1032, 1064, 1033, 1065, 1034, 1066, 1035, 1067, 1036, 1068, 1037, 1069, 1038, 1070, 1039, 1071, 1040, 1072, 1041, 1073, 1042, 1074, 1043, 1075, 1044, 1076, 1045, 1077, 1046, 1078, 1047, 1079, 1048, 1080, 1049, 1081, 1050, 1082, 1051, 1083, 1052, 1084, 1053, 1085, 1054, 1086, 1055, 1087, 1088, 1120, 1089, 1121, 1090, 1122, 1091, 1123, 1092, 1124, 1093, 1125, 1094, 1126, 1095, 1127, 1096, 1128, 1097, 1129, 1098, 1130, 1099, 1131, 1100, 1132, 1101, 1133, 1102, 1134, 1103, 1135, 1104, 1136, 1105, 1137, 1106, 1138, 1107, 1139, 1108, 1140, 1109, 1141, 1110, 1142, 1111, 1143, 1112, 1144, 1113, 1145, 1114, 1146, 1115, 1147, 1116, 1148, 1117, 1149, 1118, 1150, 1119, 1151, 1152, 1184, 1153, 1185, 1154, 1186, 1155, 1187, 1156, 1188, 1157, 1189, 1158, 1190, 1159, 1191, 1160, 1192, 1161, 1193, 1162, 1194, 1163, 1195, 1164, 1196, 1165, 1197, 1166, 1198, 1167, 1199, 1168, 1200, 1169, 1201, 1170, 1202, 1171, 1203, 1172, 1204, 1173, 1205, 1174, 1206, 1175, 1207, 1176, 1208, 1177, 1209, 1178, 1210, 1179, 1211, 1180, 1212, 1181, 1213, 1182, 1214, 1183, 1215, 1216, 1248, 1217, 1249, 1218, 1250, 1219, 1251, 1220, 1252, 1221, 1253, 1222, 1254, 1223, 1255, 1224, 1256, 1225, 1257, 1226, 1258, 1227, 1259, 1228, 1260, 1229, 1261, 1230, 1262, 1231, 1263, 1232, 1264, 1233, 1265, 1234, 1266, 1235, 1267, 1236, 1268, 1237, 1269, 1238, 1270, 1239, 1271, 1240, 1272, 1241, 1273, 1242, 1274, 1243, 1275, 1244, 1276, 1245, 1277, 1246, 1278, 1247, 1279, 1280, 1312, 1281, 1313, 1282, 1314, 1283, 1315, 1284, 1316, 1285, 1317, 1286, 1318, 1287, 1319, 1288, 1320, 1289, 1321, 1290, 1322, 1291, 1323, 1292, 1324, 1293, 1325, 1294, 1326, 1295, 1327, 1296, 1328, 1297, 1329, 1298, 1330, 1299, 1331, 1300, 1332, 1301, 1333, 1302, 1334, 1303, 1335, 1304, 1336, 1305, 1337, 1306, 1338, 1307, 1339, 1308, 1340, 1309, 1341, 1310, 1342, 1311, 1343, 1344, 1376, 1345, 1377, 1346, 1378, 1347, 1379, 1348, 1380, 1349, 1381, 1350, 1382, 1351, 1383, 1352, 1384, 1353, 1385, 1354, 1386, 1355, 1387, 1356, 1388, 1357, 1389, 1358, 1390, 1359, 1391, 1360, 1392, 1361, 1393, 1362, 1394, 1363, 1395, 1364, 1396, 1365, 1397, 1366, 1398, 1367, 1399, 1368, 1400, 1369, 1401, 1370, 1402, 1371, 1403, 1372, 1404, 1373, 1405, 1374, 1406, 1375, 1407, 1408, 1440, 1409, 1441, 1410, 1442, 1411, 1443, 1412, 1444, 1413, 1445, 1414, 1446, 1415, 1447, 1416, 1448, 1417, 1449, 1418, 1450, 1419, 1451, 1420, 1452, 1421, 1453, 1422, 1454, 1423, 1455, 1424, 1456, 1425, 1457, 1426, 1458, 1427, 1459, 1428, 1460, 1429, 1461, 1430, 1462, 1431, 1463, 1432, 1464, 1433, 1465, 1434, 1466, 1435, 1467, 1436, 1468, 1437, 1469, 1438, 1470, 1439, 1471, 1472, 1504, 1473, 1505, 1474, 1506, 1475, 1507, 1476, 1508, 1477, 1509, 1478, 1510, 1479, 1511, 1480, 1512, 1481, 1513, 1482, 1514, 1483, 1515, 1484, 1516, 1485, 1517, 1486, 1518, 1487, 1519, 1488, 1520, 1489, 1521, 1490, 1522, 1491, 1523, 1492, 1524, 1493, 1525, 1494, 1526, 1495, 1527, 1496, 1528, 1497, 1529, 1498, 1530, 1499, 1531, 1500, 1532, 1501, 1533, 1502, 1534, 1503, 1535, 1536, 1568, 1537, 1569, 1538, 1570, 1539, 1571, 1540, 1572, 1541, 1573, 1542, 1574, 1543, 1575, 1544, 1576, 1545, 1577, 1546, 1578, 1547, 1579, 1548, 1580, 1549, 1581, 1550, 1582, 1551, 1583, 1552, 1584, 1553, 1585, 1554, 1586, 1555, 1587, 1556, 1588, 1557, 1589, 1558, 1590, 1559, 1591, 1560, 1592, 1561, 1593, 1562, 1594, 1563, 1595, 1564, 1596, 1565, 1597, 1566, 1598, 1567, 1599, 1600, 1632, 1601, 1633, 1602, 1634, 1603, 1635, 1604, 1636, 1605, 1637, 1606, 1638, 1607, 1639, 1608, 1640, 1609, 1641, 1610, 1642, 1611, 1643, 1612, 1644, 1613, 1645, 1614, 1646, 1615, 1647, 1616, 1648, 1617, 1649, 1618, 1650, 1619, 1651, 1620, 1652, 1621, 1653, 1622, 1654, 1623, 1655, 1624, 1656, 1625, 1657, 1626, 1658, 1627, 1659, 1628, 1660, 1629, 1661, 1630, 1662, 1631, 1663, 1664, 1696, 1665, 1697, 1666, 1698, 1667, 1699, 1668, 1700, 1669, 1701, 1670, 1702, 1671, 1703, 1672, 1704, 1673, 1705, 1674, 1706, 1675, 1707, 1676, 1708, 1677, 1709, 1678, 1710, 1679, 1711, 1680, 1712, 1681, 1713, 1682, 1714, 1683, 1715, 1684, 1716, 1685, 1717, 1686, 1718, 1687, 1719, 1688, 1720, 1689, 1721, 1690, 1722, 1691, 1723, 1692, 1724, 1693, 1725, 1694, 1726, 1695, 1727, 1728, 1760, 1729, 1761, 1730, 1762, 1731, 1763, 1732, 1764, 1733, 1765, 1734, 1766, 1735, 1767, 1736, 1768, 1737, 1769, 1738, 1770, 1739, 1771, 1740, 1772, 1741, 1773, 1742, 1774, 1743, 1775, 1744, 1776, 1745, 1777, 1746, 1778, 1747, 1779, 1748, 1780, 1749, 1781, 1750, 1782, 1751, 1783, 1752, 1784, 1753, 1785, 1754, 1786, 1755, 1787, 1756, 1788, 1757, 1789, 1758, 1790, 1759, 1791, 1792, 1824, 1793, 1825, 1794, 1826, 1795, 1827, 1796, 1828, 1797, 1829, 1798, 1830, 1799, 1831, 1800, 1832, 1801, 1833, 1802, 1834, 1803, 1835, 1804, 1836, 1805, 1837, 1806, 1838, 1807, 1839, 1808, 1840, 1809, 1841, 1810, 1842, 1811, 1843, 1812, 1844, 1813, 1845, 1814, 1846, 1815, 1847, 1816, 1848, 1817, 1849, 1818, 1850, 1819, 1851, 1820, 1852, 1821, 1853, 1822, 1854, 1823, 1855, 1856, 1888, 1857, 1889, 1858, 1890, 1859, 1891, 1860, 1892, 1861, 1893, 1862, 1894, 1863, 1895, 1864, 1896, 1865, 1897, 1866, 1898, 1867, 1899, 1868, 1900, 1869, 1901, 1870, 1902, 1871, 1903, 1872, 1904, 1873, 1905, 1874, 1906, 1875, 1907, 1876, 1908, 1877, 1909, 1878, 1910, 1879, 1911, 1880, 1912, 1881, 1913, 1882, 1914, 1883, 1915, 1884, 1916, 1885, 1917, 1886, 1918, 1887, 1919, 1920, 1952, 1921, 1953, 1922, 1954, 1923, 1955, 1924, 1956, 1925, 1957, 1926, 1958, 1927, 1959, 1928, 1960, 1929, 1961, 1930, 1962, 1931, 1963, 1932, 1964, 1933, 1965, 1934, 1966, 1935, 1967, 1936, 1968, 1937, 1969, 1938, 1970, 1939, 1971, 1940, 1972, 1941, 1973, 1942, 1974, 1943, 1975, 1944, 1976, 1945, 1977, 1946, 1978, 1947, 1979, 1948, 1980, 1949, 1981, 1950, 1982, 1951, 1983, 1984, 2016, 1985, 2017, 1986, 2018, 1987, 2019, 1988, 2020, 1989, 2021, 1990, 2022, 1991, 2023, 1992, 2024, 1993, 2025, 1994, 2026, 1995, 2027, 1996, 2028, 1997, 2029, 1998, 2030, 1999, 2031, 2000, 2032, 2001, 2033, 2002, 2034, 2003, 2035, 2004, 2036, 2005, 2037, 2006, 2038, 2007, 2039, 2008, 2040, 2009, 2041, 2010, 2042, 2011, 2043, 2012, 2044, 2013, 2045, 2014, 2046, 2015, 2047]
        # perm_k = [0, 32, 1, 33, 2, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39, 8, 40, 9, 41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47, 16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55, 24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63, 64, 96, 65, 97, 66, 98, 67, 99, 68, 100, 69, 101, 70, 102, 71, 103, 72, 104, 73, 105, 74, 106, 75, 107, 76, 108, 77, 109, 78, 110, 79, 111, 80, 112, 81, 113, 82, 114, 83, 115, 84, 116, 85, 117, 86, 118, 87, 119, 88, 120, 89, 121, 90, 122, 91, 123, 92, 124, 93, 125, 94, 126, 95, 127, 128, 160, 129, 161, 130, 162, 131, 163, 132, 164, 133, 165, 134, 166, 135, 167, 136, 168, 137, 169, 138, 170, 139, 171, 140, 172, 141, 173, 142, 174, 143, 175, 144, 176, 145, 177, 146, 178, 147, 179, 148, 180, 149, 181, 150, 182, 151, 183, 152, 184, 153, 185, 154, 186, 155, 187, 156, 188, 157, 189, 158, 190, 159, 191, 192, 224, 193, 225, 194, 226, 195, 227, 196, 228, 197, 229, 198, 230, 199, 231, 200, 232, 201, 233, 202, 234, 203, 235, 204, 236, 205, 237, 206, 238, 207, 239, 208, 240, 209, 241, 210, 242, 211, 243, 212, 244, 213, 245, 214, 246, 215, 247, 216, 248, 217, 249, 218, 250, 219, 251, 220, 252, 221, 253, 222, 254, 223, 255, 256, 288, 257, 289, 258, 290, 259, 291, 260, 292, 261, 293, 262, 294, 263, 295, 264, 296, 265, 297, 266, 298, 267, 299, 268, 300, 269, 301, 270, 302, 271, 303, 272, 304, 273, 305, 274, 306, 275, 307, 276, 308, 277, 309, 278, 310, 279, 311, 280, 312, 281, 313, 282, 314, 283, 315, 284, 316, 285, 317, 286, 318, 287, 319, 320, 352, 321, 353, 322, 354, 323, 355, 324, 356, 325, 357, 326, 358, 327, 359, 328, 360, 329, 361, 330, 362, 331, 363, 332, 364, 333, 365, 334, 366, 335, 367, 336, 368, 337, 369, 338, 370, 339, 371, 340, 372, 341, 373, 342, 374, 343, 375, 344, 376, 345, 377, 346, 378, 347, 379, 348, 380, 349, 381, 350, 382, 351, 383, 384, 416, 385, 417, 386, 418, 387, 419, 388, 420, 389, 421, 390, 422, 391, 423, 392, 424, 393, 425, 394, 426, 395, 427, 396, 428, 397, 429, 398, 430, 399, 431, 400, 432, 401, 433, 402, 434, 403, 435, 404, 436, 405, 437, 406, 438, 407, 439, 408, 440, 409, 441, 410, 442, 411, 443, 412, 444, 413, 445, 414, 446, 415, 447, 448, 480, 449, 481, 450, 482, 451, 483, 452, 484, 453, 485, 454, 486, 455, 487, 456, 488, 457, 489, 458, 490, 459, 491, 460, 492, 461, 493, 462, 494, 463, 495, 464, 496, 465, 497, 466, 498, 467, 499, 468, 500, 469, 501, 470, 502, 471, 503, 472, 504, 473, 505, 474, 506, 475, 507, 476, 508, 477, 509, 478, 510, 479, 511]
        
        # query_states = (hidden_states @ self.q_proj.weight[perm_q].T).view(hidden_shape).transpose(1, 2)
        # key_states = (hidden_states @ self.k_proj.weight[perm_k].T).view(hidden_shape).transpose(1, 2)
        # value_states = (hidden_states @ self.v_proj.weight.T).view(hidden_shape).transpose(1, 2)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        # if self.config._attn_implementation != "eager":
        #     if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
        #         logger.warning_once(
        #             "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
        #             'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        #         )
        #     else:
        #         attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.run_attention = True
        self.run_mlp = True
        self.run_norm = True
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        if self.run_attention:
            residual = hidden_states

            if self.run_norm:
                hidden_states = self.input_layernorm(hidden_states)

            # Self Attention
            hidden_states, self_attn_weights = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = residual + hidden_states

        if self.run_mlp:
            # Fully Connected
            residual = hidden_states
            if self.run_norm:
                hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # self.layers = nn.ModuleList(
        #     [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(change_me_layers_num)]
        # )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        for decoder_layer in self.layers[:change_me_layers_num]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        # print(hidden_states[:, 2:4, 10:20])

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        # print(logits[:, -1, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
The Llama Model transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForQuestionAnswering(LlamaPreTrainedModel):
    base_model_prefix = "transformer"

    # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->Llama
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The Llama Model transformer with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForTokenClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]
