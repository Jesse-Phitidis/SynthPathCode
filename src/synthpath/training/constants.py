both_sides_to_same_map = {
    41: 2,
    42: 3,
    43: 4,
    44: 5,
    46: 7,
    47: 8,
    49: 10,
    50: 11,
    51: 12,
    52: 13,
    53: 17,
    54: 18,
    58: 26,
    60: 28
}

synthseg_and_path_to_consecutive_labels_map = {
    0: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    7: 5,
    8: 6,
    10: 7,
    11: 8,
    12: 9,
    13: 10,
    14: 11,
    15: 12,
    16: 13,
    17: 14,
    18: 15,
    24: 16,
    26: 17,
    28: 18,
    100: 19,
} # Assuming pathology has index 100

labels_to_regions_map = {
    0: "background",
    1: "cerebral_white_matter",
    2: "cerebral_cortex",
    3: "lateral_ventricle",
    4: "inferior_lateral_ventricle",
    5: "cerebellum_white_matter",
    6: "cerebellum_cortex",
    7: "thalamus",
    8: "caudate",
    9: "putamen",
    10: "pallidum",
    11: "third_ventricle",
    12: "fourth_ventricle",
    13: "brain_stem",
    14: "hippocampus",
    15: "amygdala",
    16: "csf",
    17: "accumbens_area",
    18: "ventral_dc",
    19: "stroke",
} # Assuming pathology is stroke

ordered_region_list = list(labels_to_regions_map.values())[1:]

all_possible_labels = list(labels_to_regions_map)


### INTENSITIES ###

# LR_minmax

default_intensities_mean_lr_minmax = {
    'background': 0.0,
    'cerebral_white_matter': 0.45546608244953507,
    'cerebral_cortex': 0.42730988223804856,
    'lateral_ventricle': 0.13700606341296498,
    'inferior_lateral_ventricle': 0.26067806250852404,
    'cerebellum_white_matter': 0.49391848979136765,
    'cerebellum_cortex': 0.4950450297652391,
    'thalamus': 0.4659792413430219,
    'caudate': 0.4715145247540546,
    'putamen': 0.43894216797403235,
    'pallidum': 0.34469044213549394,
    'third_ventricle': 0.18728657474963767,
    'fourth_ventricle': 0.24563495337923183,
    'brain_stem': 0.45249240111467764,
    'hippocampus': 0.5241862694977408,
    'amygdala': 0.5641582827621451,
    'csf': 0.1692627351489754,
    'accumbens_area': 0.5587710664089407,
    'ventral_dc': 0.427922460570288,
    'stroke': 0.5872477587909706
}

default_intensities_std_lr_minmax = {
    'background': 0.0,
    'cerebral_white_matter': 0.10620311631116093,
    'cerebral_cortex': 0.13050366463999705,
    'lateral_ventricle': 0.13239651457894808,
    'inferior_lateral_ventricle': 0.17305670359164244,
    'cerebellum_white_matter': 0.10809744046071822,
    'cerebellum_cortex': 0.1351130707012398,
    'thalamus': 0.11961598286407578,
    'caudate': 0.13096220979312567,
    'putamen': 0.11386674993867828,
    'pallidum': 0.09314486168883984,
    'third_ventricle': 0.13874712997660138,
    'fourth_ventricle': 0.1569403236481481,
    'brain_stem': 0.1366806331109184,
    'hippocampus': 0.14219712940139512,
    'amygdala': 0.12955091368314653,
    'csf': 0.13174789387852448,
    'accumbens_area': 0.12073798160954853,
    'ventral_dc': 0.13622015324375591,
    'stroke': 0.1718518351360231
 }

# LR minmax under

default_intensities_mean_lr_minmax_under = {
    'background': 0.0,
    'cerebral_white_matter': 0.4054660824495351,
    'cerebral_cortex': 0.37730988223804857,
    'lateral_ventricle': 0.08700606341296498,
    'inferior_lateral_ventricle': 0.21067806250852406,
    'cerebellum_white_matter': 0.44391848979136767,
    'cerebellum_cortex': 0.44504502976523913,
    'thalamus': 0.4159792413430219,
    'caudate': 0.4215145247540546,
    'putamen': 0.38894216797403236,
    'pallidum': 0.29469044213549395,
    'third_ventricle': 0.13728657474963768,
    'fourth_ventricle': 0.1956349533792318,
    'brain_stem': 0.40249240111467766,
    'hippocampus': 0.47418626949774084,
    'amygdala': 0.514158282762145,
    'csf': 0.11926273514897541,
    'accumbens_area': 0.5087710664089407,
    'ventral_dc': 0.377922460570288,
    'stroke': 0.5372477587909705
 }

default_intensities_std_lr_minmax_under = {
    'background': 0.0,
    'cerebral_white_matter': 0.05620311631116093,
    'cerebral_cortex': 0.08050366463999704,
    'lateral_ventricle': 0.08239651457894807,
    'inferior_lateral_ventricle': 0.12305670359164243,
    'cerebellum_white_matter': 0.058097440460718214,
    'cerebellum_cortex': 0.08511307070123979,
    'thalamus': 0.06961598286407578,
    'caudate': 0.08096220979312567,
    'putamen': 0.06386674993867827,
    'pallidum': 0.04314486168883984,
    'third_ventricle': 0.08874712997660138,
    'fourth_ventricle': 0.10694032364814811,
    'brain_stem': 0.0866806331109184,
    'hippocampus': 0.09219712940139511,
    'amygdala': 0.07955091368314653,
    'csf': 0.08174789387852448,
    'accumbens_area': 0.07073798160954853,
    'ventral_dc': 0.08622015324375591,
    'stroke': 0.12185183513602309
 }

# LR minmax over

default_intensities_mean_lr_minmax_over = {
    'background': 0.0,
    'cerebral_white_matter': 0.5054660824495351,
    'cerebral_cortex': 0.47730988223804854,
    'lateral_ventricle': 0.187006063412965,
    'inferior_lateral_ventricle': 0.31067806250852403,
    'cerebellum_white_matter': 0.5439184897913677,
    'cerebellum_cortex': 0.5450450297652392,
    'thalamus': 0.5159792413430219,
    'caudate': 0.5215145247540546,
    'putamen': 0.48894216797403234,
    'pallidum': 0.3946904421354939,
    'third_ventricle': 0.23728657474963766,
    'fourth_ventricle': 0.29563495337923185,
    'brain_stem': 0.5024924011146776,
    'hippocampus': 0.5741862694977409,
    'amygdala': 0.6141582827621451,
    'csf': 0.21926273514897543,
    'accumbens_area': 0.6087710664089407,
    'ventral_dc': 0.477922460570288,
    'stroke': 0.6372477587909706
}

default_intensities_std_lr_minmax_over = {
    'background': 0.0,
    'cerebral_white_matter': 0.15620311631116093,
    'cerebral_cortex': 0.18050366463999706,
    'lateral_ventricle': 0.18239651457894807,
    'inferior_lateral_ventricle': 0.22305670359164242,
    'cerebellum_white_matter': 0.15809744046071822,
    'cerebellum_cortex': 0.18511307070123978,
    'thalamus': 0.16961598286407578,
    'caudate': 0.18096220979312566,
    'putamen': 0.16386674993867828,
    'pallidum': 0.14314486168883983,
    'third_ventricle': 0.18874712997660137,
    'fourth_ventricle': 0.20694032364814813,
    'brain_stem': 0.18668063311091843,
    'hippocampus': 0.1921971294013951,
    'amygdala': 0.17955091368314652,
    'csf': 0.18174789387852447,
    'accumbens_area': 0.17073798160954853,
    'ventral_dc': 0.1862201532437559,
    'stroke': 0.22185183513602308
 }

# LR minmax dwi

dwi_default_intensities_mean_lr_minmax = {
    'background': 0.0,
    'cerebral_white_matter': 0.17616776410857024,
    'cerebral_cortex': 0.16860352450443186,
    'lateral_ventricle': 0.07831838384228657,
    'inferior_lateral_ventricle': 0.10023112685066018,
    'cerebellum_white_matter': 0.18954370608578522,
    'cerebellum_cortex': 0.2009712629041027,
    'thalamus': 0.16292688306069367,
    'caudate': 0.16150233568867128,
    'putamen': 0.15173553726282096,
    'pallidum': 0.1021657055764938,
    'third_ventricle': 0.06403476278090221,
    'fourth_ventricle': 0.0848945833351827,
    'brain_stem': 0.15015000421798552,
    'hippocampus': 0.18250096164726162,
    'amygdala': 0.2024904212685335,
    'csf': 0.07333749748488412,
    'accumbens_area': 0.18758769020058014,
    'ventral_dc': 0.13855541346569628,
    'stroke': 0.30814538156689064
}

dwi_default_intensities_std_lr_minmax = {
    'background': 0.0,
    'cerebral_white_matter': 0.05636742921165477,
    'cerebral_cortex': 0.07204270487427317,
    'lateral_ventricle': 0.03820423568115685,
    'inferior_lateral_ventricle': 0.048760945511248155,
    'cerebellum_white_matter': 0.06550752675356028,
    'cerebellum_cortex': 0.09062663684711238,
    'thalamus': 0.05324624706943574,
    'caudate': 0.05831498118513813,
    'putamen': 0.056261636778376686,
    'pallidum': 0.03837620855242647,
    'third_ventricle': 0.04225497236019908,
    'fourth_ventricle': 0.05202454415048346,
    'brain_stem': 0.06974482281868111,
    'hippocampus': 0.06771726850279172,
    'amygdala': 0.06652089080241988,
    'csf': 0.06453607487171187,
    'accumbens_area': 0.09369340642961879,
    'ventral_dc': 0.06198491094676878,
    'stroke': 0.14003201405037344
}

# LR minmax dwi under

dwi_default_intensities_mean_lr_minmax_under = {
    'background': 0.0,
    'cerebral_white_matter': 0.12616776410857022,
    'cerebral_cortex': 0.11860352450443186,
    'lateral_ventricle': 0.028318383842286562,
    'inferior_lateral_ventricle': 0.05023112685066018,
    'cerebellum_white_matter': 0.1395437060857852,
    'cerebellum_cortex': 0.15097126290410268,
    'thalamus': 0.11292688306069366,
    'caudate': 0.11150233568867128,
    'putamen': 0.10173553726282096,
    'pallidum': 0.052165705576493804,
    'third_ventricle': 0.01403476278090221,
    'fourth_ventricle': 0.034894583335182694,
    'brain_stem': 0.10015000421798552,
    'hippocampus': 0.1325009616472616,
    'amygdala': 0.15249042126853352,
    'csf': 0.023337497484884118,
    'accumbens_area': 0.13758769020058015,
    'ventral_dc': 0.08855541346569627,
    'stroke': 0.25814538156689065
 }

dwi_default_intensities_std_lr_minmax_under = {
    'background': 0.0,
    'cerebral_white_matter': 0.006367429211654764,
    'cerebral_cortex': 0.022042704874273164,
    'lateral_ventricle': 0.0,
    'inferior_lateral_ventricle': 0.0,
    'cerebellum_white_matter': 0.015507526753560277,
    'cerebellum_cortex': 0.040626636847112374,
    'thalamus': 0.0032462470694357357,
    'caudate': 0.008314981185138126,
    'putamen': 0.0062616367783766835,
    'pallidum': 0.0,
    'third_ventricle': 0.0,
    'fourth_ventricle': 0.002024544150483455,
    'brain_stem': 0.019744822818681104,
    'hippocampus': 0.01771726850279172,
    'amygdala': 0.016520890802419877,
    'csf': 0.014536074871711865,
    'accumbens_area': 0.04369340642961879,
    'ventral_dc': 0.011984910946768776,
    'stroke': 0.09003201405037344
 }

# LR minmax dwi over

dwi_default_intensities_mean_lr_minmax_over = {
    'background': 0.0,
    'cerebral_white_matter': 0.22616776410857026,
    'cerebral_cortex': 0.21860352450443188,
    'lateral_ventricle': 0.12831838384228655,
    'inferior_lateral_ventricle': 0.1502311268506602,
    'cerebellum_white_matter': 0.23954370608578524,
    'cerebellum_cortex': 0.2509712629041027,
    'thalamus': 0.21292688306069368,
    'caudate': 0.21150233568867127,
    'putamen': 0.20173553726282095,
    'pallidum': 0.1521657055764938,
    'third_ventricle': 0.11403476278090222,
    'fourth_ventricle': 0.1348945833351827,
    'brain_stem': 0.2001500042179855,
    'hippocampus': 0.23250096164726164,
    'amygdala': 0.2524904212685335,
    'csf': 0.12333749748488412,
    'accumbens_area': 0.23758769020058013,
    'ventral_dc': 0.18855541346569626,
    'stroke': 0.35814538156689063
 }

dwi_default_intensities_std_lr_minmax_over = {
    'background': 0.0,
    'cerebral_white_matter': 0.10636742921165476,
    'cerebral_cortex': 0.12204270487427317,
    'lateral_ventricle': 0.08820423568115685,
    'inferior_lateral_ventricle': 0.09876094551124816,
    'cerebellum_white_matter': 0.11550752675356028,
    'cerebellum_cortex': 0.14062663684711238,
    'thalamus': 0.10324624706943575,
    'caudate': 0.10831498118513813,
    'putamen': 0.10626163677837669,
    'pallidum': 0.08837620855242648,
    'third_ventricle': 0.09225497236019908,
    'fourth_ventricle': 0.10202454415048345,
    'brain_stem': 0.11974482281868111,
    'hippocampus': 0.11771726850279172,
    'amygdala': 0.11652089080241988,
    'csf': 0.11453607487171187,
    'accumbens_area': 0.1436934064296188,
    'ventral_dc': 0.11198491094676878,
    'stroke': 0.19003201405037345
 }


# constant std

std_const_small = {
    'background': 0.0,
    'cerebral_white_matter': 0.03,
    'cerebral_cortex': 0.03,
    'lateral_ventricle': 0.03,
    'inferior_lateral_ventricle': 0.03,
    'cerebellum_white_matter': 0.03,
    'cerebellum_cortex': 0.03,
    'thalamus': 0.03,
    'caudate': 0.03,
    'putamen': 0.03,
    'pallidum': 0.03,
    'third_ventricle': 0.03,
    'fourth_ventricle': 0.03,
    'brain_stem': 0.03,
    'hippocampus': 0.03,
    'amygdala': 0.03,
    'csf': 0.03,
    'accumbens_area': 0.03,
    'ventral_dc': 0.03,
    'stroke': 0.03
 }

std_const_medium = {
    'background': 0.0,
    'cerebral_white_matter': 0.1,
    'cerebral_cortex': 0.1,
    'lateral_ventricle': 0.1,
    'inferior_lateral_ventricle': 0.1,
    'cerebellum_white_matter': 0.1,
    'cerebellum_cortex': 0.1,
    'thalamus': 0.1,
    'caudate': 0.1,
    'putamen': 0.1,
    'pallidum': 0.1,
    'third_ventricle': 0.1,
    'fourth_ventricle': 0.1,
    'brain_stem': 0.1,
    'hippocampus': 0.1,
    'amygdala': 0.1,
    'csf': 0.1,
    'accumbens_area': 0.1,
    'ventral_dc': 0.1,
    'stroke': 0.1
 }