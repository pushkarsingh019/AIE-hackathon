# Keep Models Configurable

The literature-review workflow must stay independent of any single model provider or local model setup. Research Projects should be able to configure models for extraction support, retrieval, reranking, and answer synthesis without changing the domain workflow, because users need to mix local and hosted models as quality, cost, and availability change.

For retrieval indexing, the default behavior should prefer the strongest hosted encoder profile the project can access, because indexing can be slow or expensive if it creates better retrieval material. The user can still explicitly select a cheaper hosted provider in configuration when cost or quota matters.

For the current embedding implementation, use OpenAI embeddings by default and allow Gemini as an alternate hosted provider. Local OpenAI-compatible models remain available for chat/answer synthesis, but retrieval indexing should fail clearly if hosted embedding configuration is missing rather than silently degrading to the old local embedding endpoint.
