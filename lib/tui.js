// Raw ANSI TUI — no dependencies
const ESC = '\x1b['
const CLEAR = ESC + '2J' + ESC + 'H'
const HIDE_CURSOR = ESC + '?25l'
const SHOW_CURSOR = ESC + '?25h'
const BOLD = ESC + '1m'
const DIM = ESC + '2m'
const RESET = ESC + '0m'
const GREEN = ESC + '32m'
const YELLOW = ESC + '33m'
const CYAN = ESC + '36m'
const RED = ESC + '31m'
const MAGENTA = ESC + '35m'
const WHITE = ESC + '37m'

const SPINNER_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

function clear() { process.stdout.write(CLEAR) }

function banner() {
  const lines = [
    `${CYAN}${BOLD}╔══════════════════════════════════════╗${RESET}`,
    `${CYAN}${BOLD}║     Feature Distillation Pipeline     ║${RESET}`,
    `${CYAN}${BOLD}╚══════════════════════════════════════╝${RESET}`,
    ''
  ]
  process.stdout.write(lines.join('\n') + '\n')
}

function log(msg, color = WHITE) {
  process.stdout.write(`${color}  ${msg}${RESET}\n`)
}

function success(msg) { log(`✓ ${msg}`, GREEN) }
function warn(msg) { log(`⚠ ${msg}`, YELLOW) }
function error(msg) { log(`✗ ${msg}`, RED) }
function info(msg) { log(`→ ${msg}`, CYAN) }
function dim(msg) { log(msg, DIM) }

function header(title) {
  process.stdout.write(`\n${MAGENTA}${BOLD}  ── ${title} ──${RESET}\n\n`)
}

function progress(current, total, label = '') {
  const width = 30
  const pct = Math.min(current / total, 1)
  const filled = Math.round(width * pct)
  const empty = width - filled
  const bar = `${GREEN}${'█'.repeat(filled)}${DIM}${'░'.repeat(empty)}${RESET}`
  const pctStr = `${Math.round(pct * 100)}%`.padStart(4)
  // \r to overwrite the same line
  process.stdout.write(`\r  ${bar} ${pctStr} ${label}`)
  if (current >= total) process.stdout.write('\n')
}

function spinner(label) {
  let i = 0
  let running = true
  const id = setInterval(() => {
    if (!running) return
    process.stdout.write(`\r  ${CYAN}${SPINNER_FRAMES[i % SPINNER_FRAMES.length]}${RESET} ${label}`)
    i++
  }, 80)
  return {
    stop(finalMsg) {
      running = false
      clearInterval(id)
      process.stdout.write(`\r  ${GREEN}✓${RESET} ${finalMsg || label}\n`)
    },
    fail(finalMsg) {
      running = false
      clearInterval(id)
      process.stdout.write(`\r  ${RED}✗${RESET} ${finalMsg || label}\n`)
    }
  }
}

// Read a line from stdin
function prompt(question) {
  return new Promise(resolve => {
    process.stdout.write(`${CYAN}  ? ${RESET}${question} `)
    let buf = ''
    process.stdin.setRawMode?.(false)
    process.stdin.resume()
    process.stdin.setEncoding('utf8')
    const onData = chunk => {
      buf += chunk
      if (buf.includes('\n')) {
        process.stdin.pause()
        process.stdin.removeListener('data', onData)
        resolve(buf.trim())
      }
    }
    process.stdin.on('data', onData)
  })
}

// Show a menu, return selected index
async function menu(title, items) {
  let selected = 0

  return new Promise(resolve => {
    process.stdin.setRawMode?.(true)
    process.stdin.resume()
    process.stdin.setEncoding('utf8')

    const render = () => {
      // Move cursor up to overwrite previous render
      if (selected !== -1) {
        process.stdout.write(ESC + `${items.length + 2}A`)
      }
      process.stdout.write(`\n${CYAN}${BOLD}  ${title}${RESET}\n`)
      items.forEach((item, i) => {
        const prefix = i === selected ? `${GREEN}  ❯ ` : `${DIM}    `
        process.stdout.write(`${prefix}${item}${RESET}\n`)
      })
    }

    // Initial render — no move-up needed
    selected = -1
    selected = 0
    process.stdout.write(`\n${CYAN}${BOLD}  ${title}${RESET}\n`)
    items.forEach((item, i) => {
      const prefix = i === 0 ? `${GREEN}  ❯ ` : `${DIM}    `
      process.stdout.write(`${prefix}${item}${RESET}\n`)
    })

    const onKey = key => {
      if (key === '\x1b[A' && selected > 0) { selected--; render() }
      else if (key === '\x1b[B' && selected < items.length - 1) { selected++; render() }
      else if (key === '\r') {
        process.stdin.setRawMode?.(false)
        process.stdin.pause()
        process.stdin.removeListener('data', onKey)
        process.stdout.write('\n')
        resolve(selected)
      }
      else if (key === '\x03') { // ctrl-c
        process.stdout.write(SHOW_CURSOR)
        process.exit(0)
      }
    }
    process.stdin.on('data', onKey)
  })
}

function table(rows, headers) {
  const cols = headers.length
  const widths = headers.map((h, i) =>
    Math.max(h.length, ...rows.map(r => String(r[i] ?? '').length))
  )
  const sep = widths.map(w => '─'.repeat(w + 2)).join('┼')

  process.stdout.write(`\n  ${DIM}${sep}${RESET}\n`)
  process.stdout.write(`  ${headers.map((h, i) => `${BOLD} ${h.padEnd(widths[i])} ${RESET}`).join('│')}\n`)
  process.stdout.write(`  ${DIM}${sep}${RESET}\n`)
  for (const row of rows) {
    process.stdout.write(`  ${row.map((c, i) => ` ${String(c ?? '').padEnd(widths[i])} `).join('│')}\n`)
  }
  process.stdout.write(`  ${DIM}${sep}${RESET}\n\n`)
}

export {
  clear, banner, log, success, warn, error, info, dim,
  header, progress, spinner, prompt, menu, table,
  SHOW_CURSOR, HIDE_CURSOR
}
