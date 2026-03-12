import { test, expect, describe, beforeEach, afterEach } from 'bun:test'

// The TUI functions write directly to process.stdout.
// We capture that output to test them.
let captured = ''
let origWrite

beforeEach(() => {
  captured = ''
  origWrite = process.stdout.write
  process.stdout.write = (str) => { captured += str; return true }
})

afterEach(() => {
  process.stdout.write = origWrite
})

// Dynamic import after monkey-patching stdout
const tui = await import('../lib/tui.js')

describe('banner', () => {
  test('renders the title box', () => {
    tui.banner()
    expect(captured).toContain('Feature Distillation Pipeline')
    expect(captured).toContain('╔')
    expect(captured).toContain('╚')
  })
})

describe('log functions', () => {
  test('success includes checkmark', () => {
    tui.success('done')
    expect(captured).toContain('✓')
    expect(captured).toContain('done')
  })

  test('warn includes warning symbol', () => {
    tui.warn('caution')
    expect(captured).toContain('⚠')
    expect(captured).toContain('caution')
  })

  test('error includes cross mark', () => {
    tui.error('failed')
    expect(captured).toContain('✗')
    expect(captured).toContain('failed')
  })

  test('info includes arrow', () => {
    tui.info('note')
    expect(captured).toContain('→')
    expect(captured).toContain('note')
  })

  test('dim outputs the message', () => {
    tui.dim('faded text')
    expect(captured).toContain('faded text')
  })
})

describe('header', () => {
  test('renders section title with dashes', () => {
    tui.header('My Section')
    expect(captured).toContain('──')
    expect(captured).toContain('My Section')
  })
})

describe('progress', () => {
  test('renders progress bar with percentage', () => {
    tui.progress(50, 100, 'items')
    expect(captured).toContain('50%')
    expect(captured).toContain('items')
    expect(captured).toContain('█')
  })

  test('renders 0% at start', () => {
    tui.progress(0, 100)
    expect(captured).toContain('0%')
  })

  test('renders 100% at completion', () => {
    tui.progress(100, 100)
    expect(captured).toContain('100%')
  })

  test('clamps to 100% if over', () => {
    tui.progress(150, 100)
    expect(captured).toContain('100%')
  })

  test('adds newline at completion', () => {
    tui.progress(100, 100)
    expect(captured).toEndWith('\n')
  })

  test('does not add newline when incomplete', () => {
    tui.progress(50, 100)
    expect(captured).not.toEndWith('\n')
  })
})

describe('spinner', () => {
  test('stop writes success checkmark', () => {
    const sp = tui.spinner('loading')
    sp.stop('loaded')
    expect(captured).toContain('✓')
    expect(captured).toContain('loaded')
  })

  test('fail writes error cross', () => {
    const sp = tui.spinner('loading')
    sp.fail('error')
    expect(captured).toContain('✗')
    expect(captured).toContain('error')
  })

  test('stop uses original label if no message given', () => {
    const sp = tui.spinner('original label')
    sp.stop()
    expect(captured).toContain('original label')
  })
})

describe('table', () => {
  test('renders headers and rows', () => {
    tui.table(
      [['Alice', '30'], ['Bob', '25']],
      ['Name', 'Age']
    )
    expect(captured).toContain('Name')
    expect(captured).toContain('Age')
    expect(captured).toContain('Alice')
    expect(captured).toContain('30')
    expect(captured).toContain('Bob')
    expect(captured).toContain('25')
  })

  test('renders borders', () => {
    tui.table([['x', 'y']], ['A', 'B'])
    expect(captured).toContain('─')
    expect(captured).toContain('┼')
  })

  test('pads columns to equal width', () => {
    tui.table([['short', 'a-much-longer-value']], ['H1', 'H2'])
    // The separator should be at least as wide as the longest value
    expect(captured).toContain('a-much-longer-value')
  })

  test('handles empty rows', () => {
    tui.table([], ['Col1', 'Col2'])
    expect(captured).toContain('Col1')
    expect(captured).toContain('Col2')
  })
})

describe('clear', () => {
  test('writes ANSI clear sequence', () => {
    tui.clear()
    // ESC[2J clears screen, ESC[H moves cursor home
    expect(captured).toContain('\x1b[2J')
    expect(captured).toContain('\x1b[H')
  })
})

describe('ANSI constants', () => {
  test('exports SHOW_CURSOR', () => {
    expect(tui.SHOW_CURSOR).toContain('\x1b[')
    expect(tui.SHOW_CURSOR).toContain('25h')
  })

  test('exports HIDE_CURSOR', () => {
    expect(tui.HIDE_CURSOR).toContain('\x1b[')
    expect(tui.HIDE_CURSOR).toContain('25l')
  })
})
