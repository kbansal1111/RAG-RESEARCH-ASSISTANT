from retriever import retrieve
from llm import generate_answer

while True:

    query = input("\nAsk Question: ")

    if query.lower() == "exit":
        break

    results = retrieve(query)

    context = "\n".join(
        results["documents"][0]
    )

    answer = generate_answer(
        query,
        context
    )

    print("\nAnswer:\n")
    print(answer)