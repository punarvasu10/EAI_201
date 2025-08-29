# Vacuum Cleaner Program

def dust_mode(entry):
    if entry == 1:
        print("Normal Mode ON (for everyday dust).")
    elif entry == 2:
        print(" Power Mode ON (for small crystals/rocks).")
    elif entry == 3:
        print(" Ultra Power Mode ON (for hard particles).")
    else:
        print(" Invalid choice.")


def vacuum_control(shape):
    print(f"\n--- {shape} Vacuum Cleaner ---")
    choice = input("Type 'start' to begin cleaning: ").lower()

    if choice == "start":
        print(f" {shape} Vacuum Started Cleaning!")
        while True:
            print("\nCommands: left | right | dock | stop")
            action = input("Enter your command: ").lower()

            if action == "left":
                print(" Turning Left...")
            elif action == "right":
                print(" Turning Right...")
            elif action == "dock":
                print(" Returning to Dock...")
            elif action == "stop":
                print(" Stopping the process.")
                break
            else:
                print(" Invalid command. Try again.")
    else:
        print(" You must type 'start' to run the vacuum.")


# Main Program 
print("Choose Vacuum Shape:")
print("1. Circle")
print("2. Square")
print("3. Triangle")
print("4. Rectangle")
shape_choice = int(input("Enter shape (1/2/3/4): "))

shapes = {1: "Circle", 2: "Square", 3: "Triangle", 4: "Rectangle"}
shape = shapes.get(shape_choice, "Circle")  # default Circle if wrong input

print("\nSelect Dust Type:")
print("1. Normal Dust")
print("2. Small Crystals/Rocks")
print("3. Hard Things")
dust_choice = int(input("Enter type (1/2/3): "))

dust_mode(dust_choice)   # run dust mode
vacuum_control(shape)    # run vacuum control
