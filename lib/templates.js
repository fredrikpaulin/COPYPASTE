import { join } from 'node:path'
import { readdir } from 'node:fs/promises'

const TEMPLATES_DIR = join(import.meta.dir, '..', 'templates')

async function listTemplates() {
  try {
    const files = await readdir(TEMPLATES_DIR)
    const templates = []
    for (const file of files) {
      if (!file.endsWith('.json')) continue
      const data = await Bun.file(join(TEMPLATES_DIR, file)).json()
      templates.push({ file, ...data })
    }
    return templates
  } catch {
    return []
  }
}

async function loadTemplate(name) {
  // Try exact filename first, then with .json
  const candidates = [
    join(TEMPLATES_DIR, name),
    join(TEMPLATES_DIR, `${name}.json`)
  ]
  for (const path of candidates) {
    const file = Bun.file(path)
    if (await file.exists()) return file.json()
  }
  throw new Error(`Template not found: ${name}`)
}

export { listTemplates, loadTemplate, TEMPLATES_DIR }
