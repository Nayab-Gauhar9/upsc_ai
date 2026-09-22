# test_step3.py
from src.classifiers.llm import LangChainGroqClassifier

classifier = LangChainGroqClassifier()

test_title = "Cabinet approves reservation for women in Parliament"
test_content = (
    "The Union Cabinet has approved the Constitution amendment bill providing "
    "33 percent quota for women in Lok Sabha and state assemblies."
)

print("Classifying test article...")
result = classifier.classify(test_title, test_content)

print("\n--- CLASSIFICATION RESULT ---")
print(result)