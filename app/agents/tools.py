# -*- coding: utf-8 -*-
"""Agent 工具集：RAG 检索问答 / 运单查询(Mock) / 理赔与售后规则。
真实项目中：order 工具通过 Feign/HTTP 调用 order-server，售后工具对接理赔审核服务。

命名约定：
- search_knowledge / query_order / after_sale_rule：原始函数，内置路由与 SSE 流式直接调用；
- *_tool 变体：CrewAI Tool 包装（仅安装 crewai 后存在），供多 Agent 编排使用。
"""
import re

from app.rag.retriever import answer_with_rag, build_llm


def search_knowledge(query: str) -> str:
    """检索客服知识库并基于资料回答问题（RAG）。"""
    answer, _ = answer_with_rag(query, build_llm())
    return answer


def query_order(message: str) -> str:
    """查询运单状态与物流轨迹（Mock 数据）。"""
    match = re.search(r"\d{6,}", message)
    order_no = match.group(0) if match else "202608090001"
    return (
        f"运单 {order_no}：状态=运输中，最新节点=已到达杭州转运中心，"
        "预计明天送达。如物流信息超过 48 小时未更新，可发起催查。"
    )


def after_sale_rule(_message: str) -> str:
    """查询理赔与售后规则。"""
    return ("理赔规则：已保价运单按保价金额与损失比例赔付；未保价按运费 3 倍赔付，"
            "单票最高 1000 元；破损需在签收后 7 天内提交照片凭证；"
            "物流超过 72 小时未更新且催查确认无法找回的视为丢失。")


# CrewAI 工具版（供 Agent 编排；未安装 crewai 时为 None，系统自动降级）
search_knowledge_tool = None
query_order_tool = None
after_sale_rule_tool = None
CREW_TOOLS_READY = False
try:
    from crewai.tools import tool
    search_knowledge_tool = tool("search_knowledge")(search_knowledge)
    query_order_tool = tool("query_order")(query_order)
    after_sale_rule_tool = tool("after_sale_rule")(after_sale_rule)
    CREW_TOOLS_READY = True
except Exception:  # pragma: no cover - crewai 未安装
    CREW_TOOLS_READY = False
