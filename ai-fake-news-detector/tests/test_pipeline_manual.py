from rag.rag_pipeline import RAGPipeline


def main():
    pipeline = RAGPipeline()

    result = pipeline.run(
        "Drinking warm lemon water cures cancer."
    )

    print("\nCLAIM:")
    print(result.claim)

    print("\nVERIFICATION:")
    print(result.verification.status)

    print("\nSUMMARY:")
    print(result.verification.summary)

    print("\nEXPLANATION:")
    if result.explanation:
        print(result.explanation.text)
    else:
        print("No explanation")

    print("\nCITATIONS:")
    for citation in result.citations:
        print(citation)


if __name__ == "__main__":
    main()