# AI Assistant for Student Grading 

def calculate_grade(avg):
    if avg >= 91:
        return "A+ (Excellent)"
    elif avg >= 81:
        return "A (Excellent)"
    elif avg >= 71:
        return "B+ (Good)"
    elif avg >= 61:
        return "B (Good)"
    elif avg >= 51:
        return "C+ (Satisfactory)"
    elif avg >= 41:
        return "C (Satisfactory)"
    elif avg >= 35:
        return "D (Marginal Pass)"
    else:
        return "F (Fail - Inadequate)"


def grading_assistant():
    print("Hi this is AI Student Grading Assistant ")
    
    subjects = int(input("Enter the number of subjects: "))
    student_marks = {}
    entered_subjects = set()  
    
    for i in range(subjects):
        while True:
            subject = input(f"Enter Name of Subject {i+1} : ").strip().title()
            
            # Check if subject is valid string or not
            if not subject.replace(" ", "").isalpha():
                print("Invalid subject name! Please enter only letters.")
                continue
            
            # Check if subject already entered 
            if subject in entered_subjects:
                print("This subject has already been entered. Please enter a different subject.")
                continue
            
            # If valid and unique, add it
            entered_subjects.add(subject)
            break
        
        # Keep asking until valid marks entered
        while True:
            marks = float(input(f"Enter marks for {subject} (0–100): "))
            if 0 <= marks <= 100:
                student_marks[subject] = marks
                break
            else:
                print("Invalid input! Marks must be between 0 and 100. Please try again.")
    
    total = sum(student_marks.values())
    avg = total / subjects
    
    print("\nStudent Report Card")
    for sub, mark in student_marks.items():
        print(f"{sub}: {mark}/100")
    
    print(f"\nTotal Marks: {total}")
    print(f"Average: {avg:.2f}")
    grade = calculate_grade(avg)
    print(f"Final Grade: {grade}")

    # AI feedback
    if "F" in grade:
        print(" You need to work harder. Focus on basics and practice regularly.")
    elif "D" in grade:
        print(" You passed, but more effort is needed to improve.")
    elif "C" in grade:
        print(" Decent performance, but you can aim higher with consistency.")
    elif "C+" in grade:
        print(" Decent performance, but you can aim higher with consistency.")
    elif "B" in grade:
        print(" Good job! Keep practicing to reach excellence.")
    elif "B+" in grade:
        print(" Good job! Keep practicing to reach excellence.")
    elif "A" in grade:
        print(" Excellent work! Maintain this consistency.")
    else:
        print(" Outstanding! You are a top performer.")


# Function call
grading_assistant()