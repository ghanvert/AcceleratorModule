import inspect

class Module:
    def __init__(self):
        pass

    def training_step(self, a):
        print("Yeey!")

module = Module()

# Assuming 'module' is an object with a 'training_step' method
training_step_function = module.training_step

# Get the number of parameters the function accepts
parameters = inspect.signature(module.training_step).parameters
num_params = len([p for p in parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)])

# Call the function with appropriate arguments
if num_params == 2:
    print("running 1")
    training_step_function("a", "b")
else:
    print("running 2")
    training_step_function("a")
