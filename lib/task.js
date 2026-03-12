import { readdir } from 'node:fs/promises'
import { join } from 'node:path'

const TASKS_DIR = join(import.meta.dir, '..', 'tasks')
const SCHEMA_PATH = join(import.meta.dir, '..', 'schemas', 'task.schema.json')

let schema = null

async function loadSchema() {
  if (!schema) {
    schema = await Bun.file(SCHEMA_PATH).json()
  }
  return schema
}

// Minimal JSON Schema validator — handles the subset we use
function validate(data, schema) {
  const errors = []

  if (schema.type === 'object') {
    if (typeof data !== 'object' || data === null || Array.isArray(data)) {
      return [{ path: '', message: 'expected object' }]
    }

    // required
    for (const key of schema.required || []) {
      if (data[key] === undefined) {
        errors.push({ path: key, message: `required field missing` })
      }
    }

    // properties
    for (const [key, propSchema] of Object.entries(schema.properties || {})) {
      if (data[key] === undefined) continue
      const val = data[key]

      if (propSchema.type === 'string' && typeof val !== 'string') {
        errors.push({ path: key, message: 'expected string' })
      }
      if (propSchema.type === 'integer' && (!Number.isInteger(val))) {
        errors.push({ path: key, message: 'expected integer' })
      }
      if (propSchema.type === 'number' && typeof val !== 'number') {
        errors.push({ path: key, message: 'expected number' })
      }
      if (propSchema.enum && !propSchema.enum.includes(val)) {
        errors.push({ path: key, message: `must be one of: ${propSchema.enum.join(', ')}` })
      }
      if (propSchema.pattern && typeof val === 'string' && !new RegExp(propSchema.pattern).test(val)) {
        errors.push({ path: key, message: `must match pattern ${propSchema.pattern}` })
      }
      if (propSchema.type === 'array') {
        if (!Array.isArray(val)) {
          errors.push({ path: key, message: 'expected array' })
        } else if (propSchema.minItems && val.length < propSchema.minItems) {
          errors.push({ path: key, message: `needs at least ${propSchema.minItems} items` })
        }
      }
      if (propSchema.minimum !== undefined && val < propSchema.minimum) {
        errors.push({ path: key, message: `minimum ${propSchema.minimum}` })
      }
      if (propSchema.maximum !== undefined && val > propSchema.maximum) {
        errors.push({ path: key, message: `maximum ${propSchema.maximum}` })
      }
    }

    // allOf conditional validation
    for (const rule of schema.allOf || []) {
      if (rule.if && rule.then) {
        const condProp = Object.keys(rule.if.properties)[0]
        const condVal = rule.if.properties[condProp].const
        if (data[condProp] === condVal) {
          for (const req of rule.then.required || []) {
            if (data[req] === undefined) {
              errors.push({ path: req, message: `required when type is "${condVal}"` })
            }
          }
        }
      }
    }
  }

  return errors
}

// Apply schema defaults to a task
function applyDefaults(task, schema) {
  for (const [key, propSchema] of Object.entries(schema.properties || {})) {
    if (task[key] === undefined && propSchema.default !== undefined) {
      task[key] = propSchema.default
    }
    if (propSchema.type === 'object' && propSchema.properties && task[key]) {
      for (const [subKey, subSchema] of Object.entries(propSchema.properties)) {
        if (task[key][subKey] === undefined && subSchema.default !== undefined) {
          task[key][subKey] = subSchema.default
        }
      }
    }
  }
  return task
}

async function loadTask(name) {
  const filePath = join(TASKS_DIR, `${name}.json`)
  const file = Bun.file(filePath)
  if (!await file.exists()) {
    throw new Error(`Task file not found: ${filePath}`)
  }
  const data = await file.json()
  const s = await loadSchema()
  const errors = validate(data, s)
  if (errors.length) {
    throw new Error(`Invalid task "${name}":\n${errors.map(e => `  ${e.path}: ${e.message}`).join('\n')}`)
  }
  return applyDefaults(data, s)
}

async function listTasks() {
  try {
    const files = await readdir(TASKS_DIR)
    return files.filter(f => f.endsWith('.json')).map(f => f.replace('.json', ''))
  } catch {
    return []
  }
}

async function saveTask(task) {
  const s = await loadSchema()
  const errors = validate(task, s)
  if (errors.length) {
    throw new Error(`Invalid task:\n${errors.map(e => `  ${e.path}: ${e.message}`).join('\n')}`)
  }
  const filePath = join(TASKS_DIR, `${task.name}.json`)
  await Bun.write(filePath, JSON.stringify(task, null, 2) + '\n')
  return filePath
}

export { loadTask, listTasks, saveTask, validate, loadSchema }
