import { expect, it } from 'vitest'
import { readablePositions, routeAroundNodes } from './graphLayout'

it('folds long chains without merging nodes or moving relationships', () => {
  const nodes = Array.from({ length: 8 }, (_, i) => ({ id: String(i), x: i * 310, y: 0, width: 220, height: 104 }))
  const result = readablePositions(nodes)
  expect(Object.keys(result)).toHaveLength(8)
  expect(new Set(Object.values(result).map(p => `${p.x},${p.y}`)).size).toBe(8)
  expect(Math.max(...Object.values(result).map(p => p.x))).toBeLessThanOrEqual(620)
  expect(result['3'].y).toBeGreaterThan(result['2'].y)
  expect(result['4'].x).toBeLessThan(result['3'].x)
})

it('routes a branch around intermediate nodes instead of through them', () => {
  const boxes = [{ x: 200, y: 100, width: 220, height: 104 }, { x: 500, y: 250, width: 220, height: 104 }]
  const start = { x: 300, y: 0 }, end = { x: 600, y: 400 }
  const route = routeAroundNodes(start, end, boxes)!
  expect(route[0]).toEqual(start)
  expect(route.at(-1)).toEqual(end)
  for (let i = 1; i < route.length; i++) {
    const a = route[i - 1], b = route[i]
    expect(a.x === b.x || a.y === b.y).toBe(true)
    for (const box of boxes) {
      const intersects = a.x === b.x
        ? a.x > box.x && a.x < box.x + box.width && Math.max(a.y, b.y) > box.y && Math.min(a.y, b.y) < box.y + box.height
        : a.y > box.y && a.y < box.y + box.height && Math.max(a.x, b.x) > box.x && Math.min(a.x, b.x) < box.x + box.width
      expect(intersects).toBe(false)
    }
  }
})

it('keeps a compact layout and returns a direct clear path', () => {
  expect(readablePositions([{ id: 'a', x: 12, y: 20 }])).toEqual({ a: { x: 12, y: 20 } })
  expect(routeAroundNodes({ x: 0, y: 0 }, { x: 100, y: 0 }, [])).toEqual([{ x: 0, y: 0 }, { x: 100, y: 0 }])
})
