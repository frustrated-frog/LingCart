type Positioned = { id: string; x?: number; y?: number; width?: number; height?: number }
export type Point = { x: number; y: number }
export type Obstacle = Point & { width: number; height: number }

/** 在矩形外侧的稀疏网格寻路，避免折行后的跨层关系穿过其他节点。 */
export function routeAroundNodes(start: Point, end: Point, boxes: Obstacle[]): Point[] | null {
  const obstacles = boxes.map(b => ({ left: b.x - 12, right: b.x + b.width + 12, top: b.y - 12, bottom: b.y + b.height + 12 }))
  const xs = [...new Set([start.x, end.x, ...obstacles.flatMap(b => [b.left, b.right])])].sort((a, b) => a - b)
  const ys = [...new Set([start.y, end.y, ...obstacles.flatMap(b => [b.top, b.bottom])])].sort((a, b) => a - b)
  // 超大型图不在主线程建立无界网格。
  if (xs.length * ys.length > 20000) return null
  const key = (x: number, y: number) => y * xs.length + x
  const first = key(xs.indexOf(start.x), ys.indexOf(start.y)), last = key(xs.indexOf(end.x), ys.indexOf(end.y))
  const distance = new Map<number, number>([[first, 0]])
  const previous = new Map<number, number>()
  const open = new Set([first])
  const point = (id: number) => ({ x: xs[id % xs.length], y: ys[Math.floor(id / xs.length)] })
  const blocked = (a: Point, b: Point) => obstacles.some(o => a.x === b.x
    ? a.x > o.left && a.x < o.right && Math.max(a.y, b.y) > o.top && Math.min(a.y, b.y) < o.bottom
    : a.y > o.top && a.y < o.bottom && Math.max(a.x, b.x) > o.left && Math.min(a.x, b.x) < o.right)
  while (open.size) {
    let best = -1, score = Infinity
    for (const id of open) {
      const p = point(id), candidate = distance.get(id)! + Math.abs(p.x - end.x) + Math.abs(p.y - end.y)
      if (candidate < score) { score = candidate; best = id }
    }
    if (best === last) {
      const route = [point(last)]
      let id = last
      while (previous.has(id)) { id = previous.get(id)!; route.unshift(point(id)) }
      return route.filter((p, i) => !i || i === route.length - 1 || !((route[i - 1].x === p.x && p.x === route[i + 1].x) || (route[i - 1].y === p.y && p.y === route[i + 1].y)))
    }
    open.delete(best)
    const x = best % xs.length, y = Math.floor(best / xs.length), a = point(best)
    for (const [nx, ny] of [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]) {
      if (nx < 0 || ny < 0 || nx >= xs.length || ny >= ys.length) continue
      const id = key(nx, ny), b = point(id)
      if (blocked(a, b)) continue
      const cost = distance.get(best)! + Math.abs(a.x - b.x) + Math.abs(a.y - b.y)
      if (cost < (distance.get(id) ?? Infinity)) { distance.set(id, cost); previous.set(id, best); open.add(id) }
    }
  }
  return null
}

/** 折叠过长的分层布局，保持每层内的纵向间距；不修改任何程序关系。 */
export function readablePositions(children: Positioned[]) {
  const ranks = [...new Set(children.map(n => n.x ?? 0))].sort((a, b) => a - b)
  const maxX = Math.max(0, ...children.map(n => (n.x ?? 0) + (n.width ?? 220)))
  const maxY = Math.max(0, ...children.map(n => (n.y ?? 0) + (n.height ?? 104)))
  if (ranks.length <= 4 || maxX < maxY * 2) return Object.fromEntries(children.map(n => [n.id, { x: n.x ?? 0, y: n.y ?? 0 }]))
  const positions: Record<string, { x: number; y: number }> = {}
  let top = 0
  for (let offset = 0; offset < ranks.length; offset += 3) {
    const group = ranks.slice(offset, offset + 3)
    const members = children.filter(n => group.includes(n.x ?? 0))
    const minY = Math.min(...members.map(n => n.y ?? 0))
    const height = Math.max(...members.map(n => (n.y ?? 0) + (n.height ?? 104))) - minY
    for (const n of members) {
      const col = group.indexOf(n.x ?? 0)
      positions[n.id] = { x: (Math.floor(offset / 3) % 2 ? 2 - col : col) * 310, y: top + (n.y ?? 0) - minY }
    }
    top += height + 100
  }
  return positions
}
