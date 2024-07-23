import { collections } from "@hypermode/functions-as";
import { models } from "@hypermode/functions-as";
import { OpenAIEmbeddingsModel } from "@hypermode/models-as/models/openai/embeddings";
import { EmbeddingsModel } from "@hypermode/models-as/models/experimental/embeddings";

// These names should match the ones defined in the hypermode.json manifest file.
const openAImbeddingModelName: string = "embeddings";
const miniLMEmbeddingsModelName: string = "minilm";
const myProducts: string = "myProducts";
const searchMethod: string = "searchMethod1";

// This function takes input text and returns the vector embedding for that text.
export function openAIEmbed(text: string[]): f32[][] {
  const model = models.getModel<OpenAIEmbeddingsModel>(openAImbeddingModelName);
  const input = model.createInput(text);
  const output = model.invoke(input);
  return output.data.map<f32[]>((d) => d.embedding);
}

export function miniLMEmbed(text: string[]): f32[][] {
  const model = models.getModel<EmbeddingsModel>(miniLMEmbeddingsModelName);
  const input = model.createInput(text);
  const output = model.invoke(input);
  return output.predictions;
}

export function addProducts(name: string[], description: string[]): string[] {
  const response = collections.upsertBatch(myProducts, name, description);
  if (!response.isSuccessful) {
    throw new Error(response.error);
  }
  return response.keys;
}

export function deleteProduct(name: string): string {
  const response = collections.remove(myProducts, name);
  if (!response.isSuccessful) {
    throw new Error(response.error);
  }
  return response.status;
}

export function getProduct(name: string): string {
  return collections.getText(myProducts, name);
}

export function computeSimilarityBetweenProducts(
  name1: string,
  name2: string,
): f64 {
  return collections.computeSimilarity(myProducts, searchMethod, name1, name2)
    .score;
}

export function searchProducts(
  description: string,
  maxItems: i32,
): collections.CollectionSearchResult {
  return collections.search(
    myProducts,
    searchMethod,
    description,
    maxItems,
    true,
  );
}

export function recomputeProductIndex(): string {
  const response = collections.recomputeSearchMethod(myProducts, searchMethod);
  if (!response.isSuccessful) {
    throw new Error(response.error);
  }
  return response.status;
}
