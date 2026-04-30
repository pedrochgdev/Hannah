from vector_store import VectorStore

# Conectar a tu base de datos
db = VectorStore(db_path="./test_vectordb")

# Ver resumen completo
db.view_database_summary()

# Obtener todos los documentos
print(f"\n{'='*50}")
print("  OBTENIENDO TODOS LOS DOCUMENTOS")
print(f"{'='*50}")

# Usar el método corregido
resultados = db.get_all_documents(limit=20)

for i, (doc_id, texto, metadata) in enumerate(zip(
    resultados['ids'], 
    resultados['documents'], 
    resultados['metadatas']
)):
    print(f"\n[{i+1}] ID: {doc_id}")
    print(f"    Texto: {texto}")
    print(f"    Metadatos: {metadata}")
    print("-" * 50)

# También puedes hacer búsquedas con filtros
print(f"\n{'='*50}")
print("  BUSCANDO DOCUMENTOS POR METADATO")
print(f"{'='*50}")

# Ejemplo: buscar documentos con source = "arquitectura"
filtrados = db.collection.get(
    where={"source": "arquitectura"},
    include=["documents", "metadatas"]
)

if filtrados['documents']:
    print(f"Documentos con source='arquitectura':")
    for doc_id, texto in zip(filtrados['ids'], filtrados['documents']):
        print(f"  - {doc_id}: {texto}")
else:
    print("No se encontraron documentos con ese filtro")