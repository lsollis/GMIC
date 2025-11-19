import pickle

# Load the exam list
with open('/workspace/outputs/processed_exam_list.pkl', 'rb') as f:
    exam_list = pickle.load(f)

# Check the first exam
sample_exam = exam_list[0]
print("Exam keys:", sample_exam.keys())

# Check if 'best_center' exists
if 'best_center' in sample_exam:
    print("✅ Optimal centers found!")
    print("Sample best_center:", sample_exam['best_center'])
    
    # Verify structure - should be like:
    # {'R-CC': [(y, x)], 'R-MLO': [(y, x)], 'L-CC': [(y, x)], 'L-MLO': [(y, x)]}
    for view, centers in sample_exam['best_center'].items():
        print(f"  {view}: {centers}")
else:
    print("❌ No optimal centers found!")
    print("Available keys:", sample_exam.keys())