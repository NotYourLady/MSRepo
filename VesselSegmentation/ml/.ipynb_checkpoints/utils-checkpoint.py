def get_total_params(model):
    total_params = sum(
    param.numel() for param in model.parameters()
    )
    return(total_params)