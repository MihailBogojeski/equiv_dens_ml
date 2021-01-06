import string
import random


def get_number_of_parameters(model):
    num = 0
    for param in model.parameters():
        if param.requires_grad:
            num += param.numel()
    return num


# used for creating a "unique" id for a run (almost impossible to generate the same twice)
def generate_id(
    size=8, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits
):
    return "".join(random.SystemRandom().choice(chars) for _ in range(size))
