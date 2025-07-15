import json
import sys
import os
sys.path.append(os.path.abspath(".."))

from Aasha_chatbot import (
    first_message,
    continue_convo,
    get_emotion_label,
    is_exit_intent,
    match_faq  
)

# Load test cases
with open("test_cases_51.json", "r") as f:
    test_cases = json.load(f)

print(f"\n🧪 Running {len(test_cases)} test cases...\n")

passed = 0
failed = 0
failed_cases = []

for i, test in enumerate(test_cases, 1):
    input_text = test["input"]
    test_type = test["type"]
    print(f"🔹 Test {i}: {input_text[:60]}{'...' if len(input_text) > 60 else ''}")

    try:
        if test_type == "emotion":
            _, meta = first_message(input_text)
            actual = meta["emotion"]
            expected = test["expected_emotion"]
            if actual == expected:
                print(f"✅ Passed | Detected emotion: {actual}")
                passed += 1
            else:
                print(f"❌ Failed | Expected: {expected}, Got: {actual}")
                test["actual_emotion"] = actual
                failed_cases.append(test)
                failed += 1

        elif test_type == "faq":
            _, meta = first_message(input_text)
            actual = meta.get("is_faq", False)
            expected = test["expected_faq"]
            if actual == expected:
                print(f"✅ Passed | FAQ detected: {actual}")
                passed += 1
            else:
                print(f"❌ Failed | Expected FAQ: {expected}, Got: {actual}")
                test["actual_faq"] = actual
                failed_cases.append(test)
                failed += 1

        elif test_type == "celebration":
            _, meta = first_message(input_text)
            actual = meta.get("celebration_type")
            expected = test["expected_celebration"]
            if actual == expected:
                print(f"✅ Passed | Celebration type: {actual}")
                passed += 1
            else:
                print(f"❌ Failed | Expected: {expected}, Got: {actual}")
                test["actual_celebration"] = actual
                failed_cases.append(test)
                failed += 1

        elif test_type == "exit":
            actual = is_exit_intent(input_text)
            expected = test["expected_exit"]
            if actual == expected:
                print(f"✅ Passed | Exit detected: {actual}")
                passed += 1
            else:
                print(f"❌ Failed | Expected Exit: {expected}, Got: {actual}")
                test["actual_exit"] = actual
                failed_cases.append(test)
                failed += 1

    except Exception as e:
        print(f"❌ Error in test: {e}")
        test["error"] = str(e)
        failed_cases.append(test)
        failed += 1

# 📊 Summary
print("\n📊 Summary")
print(f"✅ Passed: {passed}")
print(f"❌ Failed: {failed}")
print(f"📈 Accuracy: {round(passed / len(test_cases) * 100, 2)}%")

# Save failed test cases
if failed_cases:
    with open("failed_cases.json", "w") as f_out:
        json.dump(failed_cases, f_out, indent=2)
    print("\n💾 Failed test cases saved to 'failed_cases.json'")

