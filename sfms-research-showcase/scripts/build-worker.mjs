import { mkdir, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'

const output = resolve('dist/server')
await mkdir(output, { recursive: true })
await writeFile(resolve(output, 'index.js'), `
export default {
  async fetch(request, env) {
    const response = await env.ASSETS.fetch(request)
    if (response.status !== 404 || !request.headers.get('accept')?.includes('text/html')) return response
    return env.ASSETS.fetch(new Request(new URL('/index.html', request.url), request))
  },
}
`, 'utf8')
