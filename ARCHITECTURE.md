# GitHub Copilot Agent: DataStage → Databricks

## ⚠️ Nota sobre la Arquitectura

Este proyecto implementa un **GitHub Copilot Agent declarativo** usando `agent.yml`.

### Archivos No Utilizados (Legacy Extension Approach)

Los siguientes archivos son de una implementación previa como "GitHub Copilot Extension" (servidor Node.js) y **NO son necesarios** para un agente declarativo:

- `src/` - Código TypeScript del servidor
- `package.json` - Dependencias Node.js
- `tsconfig.json` - Configuración TypeScript
- `copilot-extension.json` - Config de extension
- `github-app-manifest.json` - Manifest de GitHub App
- `openapi.json` - API schema

### Arquitectura Actual (Agente Declarativo)

El agente funciona mediante:

1. **`agent.yml`** - Configuración declarativa del agente
2. **`knowledge/`** - Base de conocimiento en markdown
3. **`test-artifacts/`** - Jobs .dsx de ejemplo
4. **GitHub Copilot Chat** - Interface de usuario (`@workspace`)

No requiere servidor, deployment, ni infraestructura adicional. Es completamente client-side dentro de VS Code + GitHub Copilot.

## 🚀 Uso

Ver [README.md](README.md) y [SETUP.md](SETUP.md) para instrucciones completas.

## 🔄 Migración de Extension a Agente

Si estás migrando de la versión "Extension" a "Agent":

1. ✅ Ya tienes `agent.yml` configurado
2. ✅ Knowledge base está en `knowledge/`
3. ❌ Puedes eliminar: `src/`, `package.json`, `tsconfig.json`, archivos `.json` de config
4. ✅ Usa `@workspace` en lugar de un endpoint HTTP

## 📚 Referencias

- [GitHub Copilot Agents Documentation](https://docs.github.com/en/copilot/customizing-copilot/creating-a-custom-copilot-agent)
- [Agent Configuration Schema](https://docs.github.com/en/copilot/customizing-copilot/creating-a-custom-copilot-agent#agent-configuration-schema)
