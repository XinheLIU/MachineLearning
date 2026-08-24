#!/usr/bin/env node
// Post-build: preserve the legacy Jekyll-era URLs of the MachineLearning site.
// Writes meta-refresh stubs at each old path, redirecting to the VitePress page.
import fs from 'node:fs'
import path from 'node:path'

const DIST = path.resolve('.vitepress', 'dist')
const BASE = '/MachineLearning/'

const REDIRECTS = {
  'math/math.html': '/MachineLearning/01-mathematical-foundations/',
  'math/linear-algebra.html': '/MachineLearning/02-linear-algebra/',
  'math/introduction-to-linear-algebra.html': '/MachineLearning/02-linear-algebra/2024-03-20-introduction-to-linear-algebra.html',
  'math/statistics.html': '/MachineLearning/03-statistics/',
  'classic/machine-learning-pipeline.html': '/MachineLearning/05-ml-pipeline/',
  'classic/supervised-learning.html': '/MachineLearning/07-supervised-learning-classification/',
  'classic/understanding-linear-regression.html': '/MachineLearning/06-supervised-learning-regression/2024-03-20-supervised-learning-regression.html',
  'classic/unsupervised-learning.html': '/MachineLearning/09-unsupervised-learning/',
  'deep/learning/deep-learning.html': '/MachineLearning/11-deep-learning/2020-09-13-deep-learning.html',
  'deep/learning/neural-networks-fundamentals.html': '/MachineLearning/11-deep-learning/2024-03-20-neural-networks-fundamentals.html',
  'menu/about.html': '/MachineLearning/about.html',
  'menu/contact.html': '/MachineLearning/about.html',
  'menu/categories/math.html': '/MachineLearning/',
  'menu/categories/classic.html': '/MachineLearning/',
  'menu/categories/deep-learning.html': '/MachineLearning/',
  'menu/writing.html': '/MachineLearning/',
  'page2': '/MachineLearning/',
}

for (const [from, to] of Object.entries(REDIRECTS)) {
  const file = path.join(DIST, from)
  fs.mkdirSync(path.dirname(file), { recursive: true })
  fs.writeFileSync(
    file,
    `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Redirecting…</title>
<link rel="canonical" href="${to}">
<meta http-equiv="refresh" content="0; url=${to}">
<script>location.replace(${JSON.stringify(to)})</script>
</head>
<body>
<p>This page has moved. <a href="${to}">Continue to ${to}</a>.</p>
</body>
</html>
`
  )
  console.log(`redirect stub: ${from} -> ${to}`)
}
