import tkinter as tk
from gui import TRexGUI


def main():
    root = tk.Tk()
    app = TRexGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
