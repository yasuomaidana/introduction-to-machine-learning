import torch
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def check_instalation():
    # Use a breakpoint in the code line below to debug your script.
    print(torch.__version__)
    print(torch.cuda.is_available())
    x = torch.rand(2, 3)
    print(x)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    check_instalation()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
