"""Built-in methods, imported only when selected."""

BUILTINS = {
    "autoround": "qda_quant.steps.autoround:AutoRound",
    "awq": "qda_quant.steps.awq:AWQ",
    "fisher_trace": "qda_quant.steps.token_weights:FisherTrace",
    "gptaq": "qda_quant.steps.gptaq:GPTAQ",
    "gptq": "qda_quant.steps.gptq:GPTQ",
    "gsq": "qda_quant.steps.gsq:GSQ",
    "guided": "qda_quant.steps.token_weights:Guided",
    "head": "qda_quant.steps.head:Head",
    "permute": "qda_quant.steps.permute:Permute",
    "rotate": "qda_quant.steps.rotate:Rotate",
    "rtn": "qda_quant.steps.rtn:RTN",
    "scale_kd": "qda_quant.steps.scale_kd:ScaleKD",
    "soup": "qda_quant.steps.soup:Soup",
}
