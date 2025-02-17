from langchain_core.tools import BaseTool
from pydantic import Field, BaseModel
import os
from VectorStore import VectorStore

class TariffInput(BaseModel):
    query: str = Field(..., description="Query about tariffs or customs regulations")
    country: str = Field(..., description="Country to search tariff information about")

class TariffRagTool(BaseTool):
    name: str = "TariffRagTool"
    description: str = "Searches for tariff information and customs regulations for different countries."
    args_schema: type[BaseModel] = TariffInput
    return_direct: bool = False
    
    def __init__(self):
        super().__init__()
        self._country_docs = {}
        self._vector_stores = {}
        self._discover_and_initialize()
    
    def _discover_and_initialize(self, base_path: str = "./knowledge_base/tariff/"):
        if not os.path.exists(base_path):
            print(f"Alert: base_path {base_path} doesn't exist.")
            return

        for country_dir in os.listdir(base_path):
            country_path = os.path.join(base_path, country_dir)
            if not os.path.isdir(country_path):
                continue

            country_name = country_dir.lower()
            doc_paths = []

            for file in os.listdir(country_path):
                file_path = os.path.join(country_path, file)
                if os.path.isfile(file_path) and file.lower().endswith(('.pdf')):
                    doc_paths.append(file_path)


            if doc_paths:
                if country_name not in self._vector_stores:
                    self._country_docs[country_name] = doc_paths
                    self._vector_stores[country_name] = VectorStore.create_or_load_qdrant_vector_store(doc_paths, country_name)
                else:
                    print(f"El vector store para {country_name} ya existe. No se creará de nuevo.")

    def _run(self, query: str, country: str) -> str:
        country_lower = country.lower()

        if country_lower not in self._vector_stores:
            available_countries = ", ".join(self._vector_stores.keys())
            return f"No documents loaded for {country}. Available countries: {available_countries or 'none'}"

        vector_store = self._vector_stores[country_lower]
        relevant_docs = vector_store.similarity_search(query, k=3)

        results = []
        for i, doc in enumerate(relevant_docs, 1):
            text = doc.page_content.replace("\n", " ").strip()
            country_info = doc.metadata.get('country', '')
            file_info = doc.metadata.get('source', '')
            result_text = f"Fragment {i}"
            if country_info:
                result_text += f" [{country_info}]"
            if file_info:
                result_text += f" - Source: {os.path.basename(file_info)}"
            result_text += f":\n{text}"
            results.append(result_text)

        return f"Information retrieved about tariffs and customs regulations for {country}:\n\n" + "\n\n".join(results)

    def refresh_countries(self):
        """Updates the list of countries and documents"""
        old_countries = set(self._vector_stores.keys())
        self._discover_and_initialize()
        new_countries = set(self._vector_stores.keys())

        added = new_countries - old_countries
        existing = old_countries.intersection(new_countries)

        result = f"The list of countries has been updated.\n"
        if added:
            result += f"Countries added: {', '.join(added)}.\n"
        result += f"Available countries: {', '.join(self._vector_stores.keys())}"
        return result

    def handle_object_input(self, obj: dict) -> str:
        """Asks for more details about the object passed."""
        if isinstance(obj, dict) and "query" in obj and "country" in obj:
            return f"Could you please provide more details about the query '{obj['query']}' for the country '{obj['country']}'?"
        else:
            return "Please provide a valid object with 'query' and 'country'."
