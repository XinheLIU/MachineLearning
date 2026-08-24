import { defineConfig } from 'vitepress'
import markdownItKatex from 'markdown-it-katex'
import fs from 'node:fs'
import path from 'node:path'

// VitePress book app for Machine Learning Notes.
// Renders book/src (the mdBook source) from its canonical location.
// Output keeps .html file names; README pages are rewritten to index.html for
// directory-index URLs (/MachineLearning/<chapter>/).

const read = (p) => fs.readFileSync(new URL(p, import.meta.url), 'utf8')

// --- rewrites: README.md -> index.html --------------------------------------
const rewrites = {}
function collectReadmes(dir, prefix = '') {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
    if (!entry.isDirectory()) continue
    const rel = prefix ? `${prefix}/${entry.name}` : entry.name
    const sub = path.join(dir, entry.name)
    if (fs.existsSync(path.join(sub, 'README.md'))) {
      rewrites[`${rel}/README.md`] = `${rel}/index.md`
    }
    collectReadmes(sub, rel)
  }
}
collectReadmes(new URL('../book/src', import.meta.url).pathname)
rewrites['README.md'] = 'index.md'

// --- sidebar from the mdBook SUMMARY.md --------------------------------------
function summarySidebar() {
  const lines = read('../book/src/SUMMARY.md').split(/\r?\n/)
  const groups = []
  let current = null
  const linkOf = (p) => {
    const rel = p.replace(/\.md$/, '').replace(/^\.\//, '')
    if (rel.endsWith('/README')) return `/${rel.replace(/\/README$/, '')}/`
    return `/${rel}.html`
  }
  for (const line of lines) {
    if (/^#\s+/.test(line) && !/^#\s+Summary/.test(line)) {
      current = { text: line.replace(/^#\s+/, '').trim(), items: [] }
      groups.push(current)
      continue
    }
    const item = line.match(/^\s*[-*]\s+\[([^\]]+)\]\(([^)]+)\)/)
    if (!item || !current) continue
    const [, label, target] = item
    current.items.push({ text: label.trim(), link: linkOf(target) })
  }
  return groups.filter((g) => g.items.length > 0)
}

export default defineConfig({
  title: 'Machine Learning Notes',
  description: 'Notes on AI, Business, and Technology.',
  base: '/MachineLearning/',
  srcDir: 'book/src',
  ignoreDeadLinks: true,
  srcExclude: ['SUMMARY.md', 'AGENTS.md'],
  rewrites,
  markdown: {
    config: (md) => {
      md.use(markdownItKatex, { throwOnError: false, errorColor: '#c00' })
    },
  },
  themeConfig: {
    siteTitle: 'Machine Learning Notes',
    search: { provider: 'local' },
    outline: { level: [2, 3] },
    nav: [
      { text: 'About', link: '/about.html' },
      { text: 'Resources', link: '/resources.html' },
      { text: 'Repo', link: 'https://github.com/XinheLIU/MachineLearning' },
    ],
    sidebar: summarySidebar(),
  },
})
