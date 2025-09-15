from gui import TRexGUI


def main():
    try:
        app = TRexGUI()
        app.run()
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Critical error: {e}")
    finally:
        print("T-Rex Bot terminated.")


if __name__ == "__main__":
    main()
