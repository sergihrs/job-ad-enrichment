from model2vec import StaticModel


def trial():
    embedding_model = StaticModel.from_pretrained("minishlab/potion-base-8M")

    # Make embeddings
    embeddings = embedding_model.encode(
        ["It's dangerous to go alone!", "It's a secret to everybody."]
    )
    return embeddings


if __name__ == "__main__":
    embeddings = trial()
    print(embeddings)
    print(embeddings.shape)
