from optim.scheduler import PolynomialLR


def lr_scheduler_polynomial(optimizer, poly_step_size=1, iter_warmup=0.0, iter_max=1000, poly_power=0.9, min_lr=1e-05):
    lr_scheduler = PolynomialLR(
        optimizer,
        poly_step_size,
        iter_warmup,
        iter_max,
        poly_power,
        min_lr
    )
    return lr_scheduler
