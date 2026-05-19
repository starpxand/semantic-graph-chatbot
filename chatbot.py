"""
Чат-бот для навигации по семантическому графу.
Интеграция с локальной LLM через Ollama.
"""

import json
import re
import requests
from graph_engine import SemanticGraph


# ──────────────────────────────────────────────────────────────────
#  КОНФИГУРАЦИЯ
# ──────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"   # Можно заменить: mistral, gemma2, phi3, qwen2.5

# Системный промпт — определяет поведение LLM
SYSTEM_PROMPT = """Ты — интеллектуальный ассистент для навигации по семантическому графу знаний, посвящённому Европе рубежа XVIII–XIX веков: революциям, войнам, дипломатии и ключевым личностям эпохи.

ТВОИ ЗАДАЧИ:
1. Помогать пользователю исследовать граф: находить узлы, пути, причинно-следственные связи между событиями.
2. Объяснять алгоритмы навигации: BFS (поиск в ширину), DFS (поиск в глубину), Dijkstra (кратчайший взвешенный путь).
3. Интерпретировать результаты навигации — объяснять исторический смысл найденных путей.
4. Отвечать на вопросы об исторических событиях, личностях и институтах, представленных в графе.

ГРАФ СОДЕРЖИТ СЛЕДУЮЩИЕ УЗЛЫ (ID → Название):
ФР → Французская революция, НАП → Наполеон Бонапарт, АЛ1 → Александр I,
18Б → 18 брюмера, КОН → Консульство, ИМП → Первая империя,
КОА → Антинаполеоновские коалиции, АУС → Аустерлиц, ОВ12 → Отечественная война 1812 года,
БОР → Бородинское сражение, МОС → Пожар Москвы, ВАТ → Ватерлоо,
СВЯ → Священный союз, ВКО → Венский конгресс, КУТ → Кутузов,
НЕЙ → Маршалы Наполеона, КОД → Кодекс Наполеона, КОН_Б → Континентальная блокада,
ТИЛ → Тильзитский мир, ТРА → Трафальгар, ИСП → Пиренейская война,
ЕНА → Иена-Ауэрштедт, СТО → Сто дней, ЭЛЬ → Остров Святой Елены,
БАС → Взятие Бастилии, ДЕК → Декларация прав человека,
БЛЮ → Блюхер, ВЕЛ → Веллингтон,
BFS → Поиск в ширину, DFS → Поиск в глубину, DIJ → Алгоритм Дейкстры

КАК ОТВЕЧАТЬ:
- Будь конкретным и структурированным. Используй маркированные списки.
- Если тебе передали результат алгоритма (путь, соседей, статистику) — интерпретируй его исторически.
- Объясняй ПОЧЕМУ алгоритм нашёл именно такой путь и что этот путь означает в контексте эпохи.
- Если вопрос не связан с графом или данным историческим периодом — вежливо объясни свою специализацию.
- Отвечай на русском языке.
- Не придумывай узлы или связи, которых нет в графе.

КОНТЕКСТ СИСТЕМЫ:
Граф реализован на Python. Рёбра имеют веса от 1 (сильная/прямая связь) до 5 (слабая/косвенная).
BFS находит путь с минимальным числом рёбер (кратчайшая цепочка событий).
DFS находит первый доступный путь (может быть длиннее и обходным).
Dijkstra находит путь с минимальной суммой весов рёбер (наиболее тесно связанная цепочка)."""

# Промпты для конкретных режимов
PROMPTS = {
    "path": "Пользователь запросил поиск пути в историческом графе. Вот результат алгоритма:\n{result}\n\nОбъясни этот путь: какова историческая логика этой цепочки событий/связей, почему именно эти узлы соединены, и чем отличается использованный алгоритм ({algo}) от других.",
    "neighbors": "Пользователь изучает узел '{node}'. Вот его соседи:\n{result}\n\nОбъясни исторические связи этого узла с соседями. Какую роль играет '{node}' в этой сети событий и личностей?",
    "info": "Пользователь запрашивает информацию об узле '{node}':\n{result}\n\nДай подробное историческое объяснение этого события или личности и их места в эпохе.",
    "stats": "Вот статистика исторического графа:\n{result}\n\nПрокомментируй: почему самые связные узлы столь важны для понимания данного периода?",
    "compare": "Пользователь хочет сравнить алгоритмы BFS, DFS и Dijkstra.\nВот пути:\nBFS: {bfs}\nDFS: {dfs}\nDijkstra: {dij}\n\nСравни алгоритмы: их результаты, длины путей, исторический смысл найденных цепочек, подходящие сценарии использования.",
}


# ──────────────────────────────────────────────────────────────────
#  КЛАСС ЧАТ-БОТА
# ──────────────────────────────────────────────────────────────────

class SemanticGraphChatbot:
    """Чат-бот с интеграцией Ollama для навигации по семантическому графу."""

    def __init__(self, model: str = OLLAMA_MODEL):
        self.graph = SemanticGraph()
        self.model = model
        self.history = []   # История диалога (контекст)
        self.ollama_available = self._check_ollama()

    def _check_ollama(self) -> bool:
        """Проверить доступность Ollama."""
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                print(f"✅ Ollama запущена. Доступные модели: {', '.join(models) or 'нет'}")
                return True
        except Exception:
            pass
        print("⚠️  Ollama недоступна. Бот будет работать без LLM (только граф-функции).")
        return False

    # ──────────────────────────────
    #  Вызов Ollama API
    # ──────────────────────────────
    def _ask_llm(self, user_message: str, extra_context: str = "") -> str:
        """Отправить запрос к LLM через Ollama."""
        if not self.ollama_available:
            return "[LLM недоступна — результат графа показан выше]"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for msg in self.history[-6:]:
            messages.append(msg)

        full_message = user_message
        if extra_context:
            full_message = f"{extra_context}\n\nВопрос пользователя: {user_message}"

        messages.append({"role": "user", "content": full_message})

        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_ctx": 4096,
                        "repeat_penalty": 1.1
                    }
                },
                timeout=120
            )
            if resp.status_code == 200:
                return resp.json()["message"]["content"]
            else:
                return f"[Ошибка Ollama: {resp.status_code}]"
        except requests.exceptions.Timeout:
            return "[Превышено время ожидания ответа от LLM]"
        except Exception as e:
            return f"[Ошибка соединения с Ollama: {e}]"

    # ──────────────────────────────
    #  Разбор команд пользователя
    # ──────────────────────────────
    def _parse_intent(self, text: str) -> dict:
        """Определить намерение пользователя из текста."""
        t = text.lower().strip()

        path_patterns = [
            r"путь.*от.*до", r"найди.*путь", r"как.*добраться",
            r"связь.*между", r"соединить", r"bfs|dfs|дейкстра|dijkstra"
        ]
        for p in path_patterns:
            if re.search(p, t):
                return {"intent": "path", "text": text}

        info_patterns = [r"что такое|расскажи о|информация о|опиши|что значит|кто такой|кто такая"]
        for p in info_patterns:
            if re.search(p, t):
                return {"intent": "info", "text": text}

        neighbor_patterns = [r"сосед|связан с|смежн|рядом с|покажи связи|что связано"]
        for p in neighbor_patterns:
            if re.search(p, t):
                return {"intent": "neighbors", "text": text}

        stats_patterns = [r"статистик|сколько узл|граф в целом|общая информация|самый связн"]
        for p in stats_patterns:
            if re.search(p, t):
                return {"intent": "stats", "text": text}

        compare_patterns = [r"сравни.*алгоритм|разница.*bfs.*dfs|чем отличаются"]
        for p in compare_patterns:
            if re.search(p, t):
                return {"intent": "compare", "text": text}

        list_patterns = [r"список узл|все узл|покажи узл|что есть в граф"]
        for p in list_patterns:
            if re.search(p, t):
                return {"intent": "list", "text": text}

        return {"intent": "general", "text": text}

    def _extract_nodes(self, text: str) -> list:
        """Извлечь упомянутые узлы из текста пользователя."""
        found = []
        for nid in self.graph.get_all_node_ids():
            if nid.lower() in text.lower():
                found.append(nid)
        if not found:
            for nid, data in self.graph.nodes.items():
                label_words = data["label"].lower().split()
                for word in label_words:
                    if len(word) > 3 and word in text.lower():
                        if nid not in found:
                            found.append(nid)
        return found[:2]

    # ──────────────────────────────
    #  Обработка команд
    # ──────────────────────────────
    def _handle_path(self, text: str, nodes: list) -> str:
        if len(nodes) < 2:
            return (
                "🔍 Для поиска пути мне нужны два узла.\n"
                "Пример: «найди путь от ФР до ВКО»\n\n"
                f"Доступные узлы: {', '.join(self.graph.get_all_node_ids())}"
            )

        start, end = nodes[0], nodes[1]
        algo = "BFS"
        if "dfs" in text.lower():
            algo = "DFS"
        elif "дейкстра" in text.lower() or "dijkstra" in text.lower() or "вес" in text.lower():
            algo = "Dijkstra"

        if algo == "BFS":
            result = self.graph.bfs_path(start, end)
        elif algo == "DFS":
            result = self.graph.dfs_path(start, end)
        else:
            result = self.graph.dijkstra_path(start, end)

        path_text = self.graph.format_path_text(result)

        context = PROMPTS["path"].format(result=path_text, algo=algo)
        llm_response = self._ask_llm(text, context)

        return f"{path_text}\n\n🤖 Интерпретация LLM:\n{llm_response}"

    def _handle_neighbors(self, text: str, nodes: list) -> str:
        if not nodes:
            return "Укажите узел для просмотра связей. Например: «покажи связи узла НАП»"

        node_id = nodes[0]
        result = self.graph.get_neighbors(node_id)

        if not result["found"]:
            return result["error"]

        neighbors = result["neighbors"]
        node_label = self.graph.nodes[node_id]["label"]

        lines = [f"🔗 Связи узла [{node_id}] «{node_label}» ({len(neighbors)} связей):\n"]
        for n in neighbors:
            lines.append(f"  • [{n['id']}] {n['label']}  —({n['relation']})→  вес: {n['weight']}")

        graph_text = "\n".join(lines)
        context = PROMPTS["neighbors"].format(node=node_label, result=graph_text)
        llm_response = self._ask_llm(text, context)

        return f"{graph_text}\n\n🤖 Комментарий LLM:\n{llm_response}"

    def _handle_info(self, text: str, nodes: list) -> str:
        if not nodes:
            return "Укажите узел. Например: «что такое АУС» или «кто такой КУТ»"

        node_id = nodes[0]
        result = self.graph.get_node_info(node_id)

        if not result["found"]:
            return result["error"]

        lines = [
            f"📌 Узел: [{result['id']}] {result['label']}",
            f"   Категория: {result['category']}",
            f"   Описание: {result['description']}",
            f"   Степень (кол-во связей): {result['degree']}",
            f"   Связанные узлы: {', '.join([n['label'] for n in result['neighbors'][:5]])}"
        ]
        info_text = "\n".join(lines)
        context = PROMPTS["info"].format(node=result["label"], result=info_text)
        llm_response = self._ask_llm(text, context)

        return f"{info_text}\n\n🤖 Пояснение LLM:\n{llm_response}"

    def _handle_stats(self, text: str) -> str:
        stats = self.graph.get_stats()
        lines = [
            f"📊 Статистика графа:",
            f"   Узлов: {stats['total_nodes']}",
            f"   Рёбер: {stats['total_edges']}",
            f"   Категории: {json.dumps(stats['categories'], ensure_ascii=False)}",
            f"\n   Топ-5 наиболее связных узлов:"
        ]
        for n in stats["most_connected"]:
            lines.append(f"   • [{n['id']}] {n['label']} — {n['degree']} связей")

        stats_text = "\n".join(lines)
        context = PROMPTS["stats"].format(result=stats_text)
        llm_response = self._ask_llm(text, context)

        return f"{stats_text}\n\n🤖 Анализ LLM:\n{llm_response}"

    def _handle_compare(self, text: str, nodes: list) -> str:
        if len(nodes) < 2:
            nodes = ["ФР", "ВКО"]  # Дефолтные узлы для демонстрации

        start, end = nodes[0], nodes[1]
        bfs = self.graph.format_path_text(self.graph.bfs_path(start, end))
        dfs = self.graph.format_path_text(self.graph.dfs_path(start, end))
        dij = self.graph.format_path_text(self.graph.dijkstra_path(start, end))

        context = PROMPTS["compare"].format(bfs=bfs, dfs=dfs, dij=dij)
        llm_response = self._ask_llm(text, context)

        return (
            f"🔄 Сравнение алгоритмов (путь от [{start}] до [{end}]):\n\n"
            f"--- BFS ---\n{bfs}\n\n"
            f"--- DFS ---\n{dfs}\n\n"
            f"--- Dijkstra ---\n{dij}\n\n"
            f"🤖 Сравнение от LLM:\n{llm_response}"
        )

    def _handle_list(self, text: str) -> str:
        lines = ["📋 Все узлы графа:\n"]
        cats = {}
        for nid, data in self.graph.nodes.items():
            cat = data["category"]
            if cat not in cats:
                cats[cat] = []
            cats[cat].append(f"[{nid}] {data['label']}")

        for cat, items in sorted(cats.items()):
            lines.append(f"  {cat.upper()}:")
            for item in items:
                lines.append(f"    • {item}")
            lines.append("")

        return "\n".join(lines)

    # ──────────────────────────────
    #  Главный метод обработки
    # ──────────────────────────────
    def process(self, user_input: str) -> str:
        """Обработать сообщение пользователя и вернуть ответ."""
        user_input = user_input.strip()
        if not user_input:
            return "Введите ваш вопрос."

        self.history.append({"role": "user", "content": user_input})

        intent_data = self._parse_intent(user_input)
        intent = intent_data["intent"]

        nodes = self._extract_nodes(user_input)

        if intent == "path" or (len(nodes) >= 2 and any(
            kw in user_input.lower() for kw in ["путь", "от", "до", "соединить"]
        )):
            response = self._handle_path(user_input, nodes)
        elif intent == "neighbors":
            response = self._handle_neighbors(user_input, nodes)
        elif intent == "info":
            response = self._handle_info(user_input, nodes)
        elif intent == "stats":
            response = self._handle_stats(user_input)
        elif intent == "compare":
            response = self._handle_compare(user_input, nodes)
        elif intent == "list":
            response = self._handle_list(user_input)
        else:
            response = self._ask_llm(user_input)

        self.history.append({"role": "assistant", "content": response})

        return response

    def clear_history(self):
        """Очистить историю диалога."""
        self.history = []
        return "История диалога очищена."
