"""
Движок семантического графа.
Реализует алгоритмы навигации: BFS, DFS, Dijkstra.
Предоставляет методы анализа графа для чат-бота.
"""

from collections import defaultdict, deque
import heapq
from graph_data import NODES, EDGES


class SemanticGraph:
    """Семантический граф с алгоритмами навигации."""

    def __init__(self):
        self.nodes = NODES
        self.adj = defaultdict(list)   # смежность: узел -> [(сосед, тип, вес)]
        self.adj_rev = defaultdict(list)
        self._build_graph()

    def _build_graph(self):
        for u, v, rel_type, weight in EDGES:
            self.adj[u].append((v, rel_type, weight))
            self.adj_rev[v].append((u, rel_type, weight))
            # Граф ненаправленный для навигации
            self.adj[v].append((u, rel_type, weight))

    # ──────────────────────────────────────────────
    #  Алгоритм BFS — кратчайший путь по рёбрам
    # ──────────────────────────────────────────────
    def bfs_path(self, start: str, end: str) -> dict:
        """
        Поиск кратчайшего пути между двумя узлами (BFS).
        Возвращает: {path, edges, steps, found}
        """
        if start not in self.nodes:
            return {"found": False, "error": f"Узел '{start}' не найден в графе."}
        if end not in self.nodes:
            return {"found": False, "error": f"Узел '{end}' не найден в графе."}
        if start == end:
            return {"found": True, "path": [start], "edges": [], "steps": 0}

        visited = {start}
        queue = deque([(start, [start], [])])  # (текущий, путь, рёбра)

        while queue:
            current, path, edges = queue.popleft()
            for neighbor, rel_type, weight in self.adj[current]:
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    new_edges = edges + [(current, neighbor, rel_type, weight)]
                    if neighbor == end:
                        return {
                            "found": True,
                            "path": new_path,
                            "edges": new_edges,
                            "steps": len(new_path) - 1,
                            "algorithm": "BFS"
                        }
                    visited.add(neighbor)
                    queue.append((neighbor, new_path, new_edges))

        return {"found": False, "error": f"Пути от '{start}' до '{end}' не существует."}

    # ──────────────────────────────────────────────
    #  Алгоритм DFS — поиск в глубину
    # ──────────────────────────────────────────────
    def dfs_path(self, start: str, end: str, max_depth: int = 10) -> dict:
        """
        Поиск пути между узлами через DFS (первый найденный).
        Возвращает: {path, edges, steps, found}
        """
        if start not in self.nodes:
            return {"found": False, "error": f"Узел '{start}' не найден."}
        if end not in self.nodes:
            return {"found": False, "error": f"Узел '{end}' не найден."}

        def dfs_rec(current, target, visited, path, edges, depth):
            if depth > max_depth:
                return None
            if current == target:
                return (path[:], edges[:])
            for neighbor, rel_type, weight in self.adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    edges.append((current, neighbor, rel_type, weight))
                    result = dfs_rec(neighbor, target, visited, path, edges, depth + 1)
                    if result:
                        return result
                    path.pop()
                    edges.pop()
                    visited.discard(neighbor)
            return None

        visited = {start}
        result = dfs_rec(start, end, visited, [start], [], 0)
        if result:
            path, edges = result
            return {
                "found": True,
                "path": path,
                "edges": edges,
                "steps": len(path) - 1,
                "algorithm": "DFS"
            }
        return {"found": False, "error": f"Пути от '{start}' до '{end}' не найдено (DFS)."}

    # ──────────────────────────────────────────────
    #  Алгоритм Дейкстры — кратчайший по весам
    # ──────────────────────────────────────────────
    def dijkstra_path(self, start: str, end: str) -> dict:
        """
        Кратчайший путь по взвешенным рёбрам (Dijkstra).
        """
        if start not in self.nodes:
            return {"found": False, "error": f"Узел '{start}' не найден."}
        if end not in self.nodes:
            return {"found": False, "error": f"Узел '{end}' не найден."}

        dist = {node: float('inf') for node in self.nodes}
        dist[start] = 0
        prev = {node: None for node in self.nodes}
        prev_edge = {node: None for node in self.nodes}
        heap = [(0, start)]

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if u == end:
                break
            for v, rel_type, weight in self.adj[u]:
                nd = dist[u] + weight
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    prev_edge[v] = (u, v, rel_type, weight)
                    heapq.heappush(heap, (nd, v))

        if dist[end] == float('inf'):
            return {"found": False, "error": f"Пути от '{start}' до '{end}' не найдено."}

        # Восстановление пути
        path, edges = [], []
        cur = end
        while cur:
            path.append(cur)
            if prev_edge[cur]:
                edges.append(prev_edge[cur])
            cur = prev[cur]
        path.reverse()
        edges.reverse()

        return {
            "found": True,
            "path": path,
            "edges": edges,
            "steps": len(path) - 1,
            "total_weight": dist[end],
            "algorithm": "Dijkstra"
        }

    # ──────────────────────────────────────────────
    #  Вспомогательные методы
    # ──────────────────────────────────────────────
    def get_neighbors(self, node_id: str) -> dict:
        """Получить всех соседей узла."""
        if node_id not in self.nodes:
            return {"found": False, "error": f"Узел '{node_id}' не найден."}
        neighbors = []
        for neighbor, rel_type, weight in self.adj[node_id]:
            if neighbor in self.nodes:
                neighbors.append({
                    "id": neighbor,
                    "label": self.nodes[neighbor]["label"],
                    "relation": rel_type,
                    "weight": weight
                })
        # Убрать дубликаты
        seen = set()
        unique = []
        for n in neighbors:
            if n["id"] not in seen:
                seen.add(n["id"])
                unique.append(n)
        return {"found": True, "node": node_id, "neighbors": unique}

    def get_node_info(self, node_id: str) -> dict:
        """Получить информацию об узле."""
        if node_id not in self.nodes:
            return {"found": False, "error": f"Узел '{node_id}' не найден."}
        node = self.nodes[node_id]
        neighbors = self.get_neighbors(node_id)["neighbors"]
        return {
            "found": True,
            "id": node_id,
            "label": node["label"],
            "description": node["description"],
            "category": node["category"],
            "degree": len(neighbors),
            "neighbors": neighbors
        }

    def get_stats(self) -> dict:
        """Статистика графа."""
        degrees = {n: len(set(nb for nb, _, _ in self.adj[n])) for n in self.nodes}
        top = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(EDGES),
            "most_connected": [
                {"id": n, "label": self.nodes[n]["label"], "degree": d}
                for n, d in top
            ],
            "categories": self._count_categories()
        }

    def _count_categories(self) -> dict:
        cats = defaultdict(int)
        for n in self.nodes.values():
            cats[n["category"]] += 1
        return dict(cats)

    def find_node_by_name(self, name: str) -> str | None:
        """Найти ID узла по части его названия (регистронезависимо)."""
        name_lower = name.lower()
        if name in self.nodes:
            return name
        for nid, data in self.nodes.items():
            if name_lower in data["label"].lower():
                return nid
            if name_lower in nid.lower():
                return nid
        return None

    def format_path_text(self, result: dict) -> str:
        """Форматировать результат пути в читаемый текст."""
        if not result["found"]:
            return result.get("error", "Путь не найден.")

        path = result["path"]
        edges = result["edges"]
        algo = result.get("algorithm", "?")
        steps = result["steps"]

        lines = [f"✅ Путь найден алгоритмом {algo} ({steps} шаг(ов)):"]
        lines.append("")

        for i, node_id in enumerate(path):
            label = self.nodes[node_id]["label"]
            if i < len(edges):
                _, _, rel, _ = edges[i]
                lines.append(f"  [{node_id}] {label}")
                lines.append(f"       ↓ ({rel})")
            else:
                lines.append(f"  [{node_id}] {label}")

        if "total_weight" in result:
            lines.append(f"\n  Суммарный вес пути: {result['total_weight']}")

        return "\n".join(lines)

    def get_all_node_ids(self) -> list:
        return list(self.nodes.keys())
