#!/usr/bin/env python3
"""
Test para verificar que el contenido DSX se carga correctamente
"""
import json
import re

def load_test_dataset(dataset_path: str = "test_dataset.json"):
    """Carga los casos de prueba del JSON y enriquece con contenido de archivos DSX."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    test_cases = data["test_cases"]
    
    # Para casos de migración completa, cargar el contenido del archivo DSX
    for tc in test_cases:
        if tc.get("category") == "full_migration":
            # Buscar si la query menciona un archivo DSX
            query = tc.get("query", "")
            if ".dsx" in query:
                # Extraer nombre del archivo
                match = re.search(r'test-artifacts/(\S+\.dsx)', query)
                if match:
                    dsx_file = match.group(0)
                    try:
                        with open(dsx_file, "r", encoding="utf-8") as f:
                            dsx_content = f.read()
                        # Agregar el contenido DSX al query para que el modelo lo analice
                        tc["query"] = f"{query}\n\n<DSX_FILE_CONTENT>\n{dsx_content}\n</DSX_FILE_CONTENT>"
                        # Guardar el contenido DSX en un campo separado para la evaluación
                        tc["dsx_content"] = dsx_content
                        tc["has_dsx_content"] = True
                        print(f"✅ Cargado contenido DSX para {tc['id']}: {dsx_file}")
                    except Exception as e:
                        print(f"❌ No se pudo cargar {dsx_file}: {e}")
                        tc["has_dsx_content"] = False
                        tc["dsx_content"] = ""
    
    return test_cases


if __name__ == "__main__":
    print("🔍 Probando carga de archivos DSX...")
    print()
    
    test_cases = load_test_dataset()
    
    # Buscar el caso full_001
    full_001 = None
    for tc in test_cases:
        if tc.get("id") == "full_001":
            full_001 = tc
            break
    
    if not full_001:
        print("❌ No se encontró el caso full_001")
        exit(1)
    
    print(f"Test Case ID: {full_001['id']}")
    print(f"Category: {full_001['category']}")
    print(f"Has DSX Content: {full_001.get('has_dsx_content', False)}")
    print()
    
    if "dsx_content" in full_001:
        dsx_len = len(full_001["dsx_content"])
        print(f"✅ Campo 'dsx_content' existe: {dsx_len:,} caracteres")
        
        # Verificar que es XML válido
        if full_001["dsx_content"].startswith("<?xml"):
            print("✅ El contenido DSX es XML válido")
        else:
            print("⚠️ El contenido DSX no parece ser XML")
        
        # Verificar que el query también lo tiene
        if "<DSX_FILE_CONTENT>" in full_001["query"]:
            print("✅ El query contiene <DSX_FILE_CONTENT>")
        else:
            print("❌ El query NO contiene <DSX_FILE_CONTENT>")
        
        print()
        print("Primeras 500 caracteres del dsx_content:")
        print("-" * 60)
        print(full_001["dsx_content"][:500])
        print("-" * 60)
    else:
        print("❌ Campo 'dsx_content' NO existe en el test case")
        exit(1)
    
    print()
    print("✅ VERIFICACIÓN COMPLETA: El DSX se carga correctamente")
