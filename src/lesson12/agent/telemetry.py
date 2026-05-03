"""
遥测系统 - 第十二课引入

智能体运行时可观测性系统。

职责：
  - Span：追踪中的单个操作（LLM 调用、工具执行、记忆操作等）
  - Metrics：聚合指标（调用次数、成功率、延迟）
  - Telemetry：JSONL 日志、追踪 ID 链接、指标累积

设计原则：
  - 结构化日志：JSONL 格式，每行一个 Span，可搜索可解析
  - 追踪链接：同一交互的所有 Span 共享 trace_id
  - 指标累积：边走边统计，一目了然
"""

import json
import time
from datetime import datetime
from uuid import uuid4
from dataclasses import dataclass, asdict
from typing import Optional, Union


# ==================== 数据结构 ====================


@dataclass
class Span:
    """追踪中的单个操作。

    Attributes:
        span_id: 跨度唯一 ID
        trace_id: 所属追踪 ID
        event_type: 事件类型（llm_call / tool_request / tool_execution / memory_op / decision）
        timestamp: ISO 格式时间戳
        duration_ms: 耗时（毫秒）
        data: 事件相关数据
        error: 错误信息（如果失败）
    """
    span_id: str
    trace_id: str
    event_type: str
    timestamp: str
    duration_ms: Optional[float] = None
    data: Optional[dict] = None
    error: Optional[str] = None


@dataclass
class Metrics:
    """智能体聚合指标。

    Properties:
        avg_latency_ms: 平均 LLM 延迟
        llm_success_rate: LLM 调用成功率
        tool_success_rate: 工具调用成功率
    """
    llm_calls: int = 0
    llm_failures: int = 0
    llm_retries: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    memory_ops: int = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        """平均 LLM 延迟（毫秒）。"""
        return self.total_latency_ms / self.llm_calls if self.llm_calls > 0 else 0.0

    @property
    def llm_success_rate(self) -> float:
        """LLM 调用成功率（0.0 ~ 1.0）。"""
        return 1.0 - (self.llm_failures / self.llm_calls) if self.llm_calls > 0 else 0.0

    @property
    def tool_success_rate(self) -> float:
        """工具调用成功率（0.0 ~ 1.0）。"""
        return 1.0 - (self.tool_failures / self.tool_calls) if self.tool_calls > 0 else 0.0


# ==================== Telemetry 类 ====================


class Telemetry:
    """用于智能体可观测性的遥测系统。

    用法：
        telemetry = Telemetry("agent_telemetry.jsonl")
        trace_id = telemetry.start_trace()

        # 记录 LLM 调用
        telemetry.log_llm_call(prompt_length=100, response_length=50, duration_ms=1500)

        # 记录工具调用
        telemetry.log_tool_request("calculator", {"a": 1, "b": 2})

        # 查看指标摘要
        telemetry.print_summary()

        # 调试特定追踪
        spans = telemetry.get_trace(trace_id)
    """

    def __init__(self, log_file: str = "agent_telemetry.jsonl"):
        self.log_file = log_file
        self.current_trace_id: Optional[str] = None
        self.metrics = Metrics()
        self._spans: list[Span] = []  # 内存跨度缓存

    # ==================== 追踪管理 ====================

    def start_trace(self) -> str:
        """开始新追踪（一次完整的智能体交互）。

        Returns:
            新的 trace_id（8 位 UUID 前缀）
        """
        self.current_trace_id = str(uuid4())[:8]
        return self.current_trace_id

    # ==================== 记录方法 ====================

    def _write_span(self, span: Span) -> None:
        """写入跨度到日志文件并更新内存缓存。"""
        self._spans.append(span)

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(span), ensure_ascii=False) + "\n")

    def _make_span(self, event_type: str, duration_ms: float,
                   data: Optional[dict] = None,
                   error: Optional[str] = None) -> Span:
        """创建一个新的 Span。"""
        return Span(
            span_id=str(uuid4())[:8],
            trace_id=self.current_trace_id or "no-trace",
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            duration_ms=round(duration_ms, 2),
            data=data,
            error=error,
        )

    def log_llm_call(self, prompt_length: int, response_length: int,
                     duration_ms: float, success: bool = True,
                     retries: int = 0, error: Optional[str] = None) -> None:
        """记录 LLM 调用。

        Args:
            prompt_length: 提示词长度（字符数）
            response_length: 响应长度（字符数）
            duration_ms: 耗时（毫秒）
            success: 是否成功（JSON 解析是否成功）
            retries: 重试次数
            error: 错误信息
        """
        span = self._make_span(
            event_type="llm_call",
            duration_ms=duration_ms,
            data={
                "prompt_length": prompt_length,
                "response_length": response_length,
                "success": success,
                "retries": retries,
            },
            error=error,
        )

        self._write_span(span)

        # 更新指标
        self.metrics.llm_calls += 1
        self.metrics.total_latency_ms += duration_ms
        self.metrics.llm_retries += retries
        if not success:
            self.metrics.llm_failures += 1

    def log_tool_request(self, tool_name: str, arguments: dict,
                         duration_ms: float = 0.0,
                         error: Optional[str] = None) -> None:
        """记录工具请求（AI 选择了什么工具和参数）。

        Args:
            tool_name: 工具名称
            arguments: 参数
            duration_ms: 耗时
            error: 错误信息
        """
        span = self._make_span(
            event_type="tool_request",
            duration_ms=duration_ms,
            data={
                "tool": tool_name,
                "arguments": arguments,
            },
            error=error,
        )

        self._write_span(span)

    def log_tool_execution(self, tool_name: str, result: Union[str, dict],
                           duration_ms: float, success: bool = True,
                           error: Optional[str] = None) -> None:
        """记录工具执行（工具实际运行的结果）。

        Args:
            tool_name: 工具名称
            result: 执行结果
            duration_ms: 耗时
            success: 是否成功
            error: 错误信息
        """
        span = self._make_span(
            event_type="tool_execution",
            duration_ms=duration_ms,
            data={
                "tool": tool_name,
                "result": str(result)[:200] if result else None,
                "success": success,
            },
            error=error,
        )

        self._write_span(span)

        # 更新指标
        self.metrics.tool_calls += 1
        if not success:
            self.metrics.tool_failures += 1

    def log_memory_op(self, operation: str, data: Optional[str] = None,
                      duration_ms: float = 0.0) -> None:
        """记录记忆操作。

        Args:
            operation: 操作类型（add / search / remove / clear）
            data: 操作的数据
            duration_ms: 耗时
        """
        span = self._make_span(
            event_type="memory_op",
            duration_ms=duration_ms,
            data={
                "operation": operation,
                "data": str(data)[:200] if data else None,
            },
        )

        self._write_span(span)

        self.metrics.memory_ops += 1

    def log_decision(self, choices: list[str], selected: str,
                     duration_ms: float = 0.0,
                     error: Optional[str] = None) -> None:
        """记录决策（第四课意图识别）。

        Args:
            choices: 可选列表
            selected: 选中的选项
            duration_ms: 耗时
            error: 错误信息
        """
        span = self._make_span(
            event_type="decision",
            duration_ms=duration_ms,
            data={
                "choices": choices[:10],  # 截断防止日志过大
                "selected": selected,
            },
            error=error,
        )

        self._write_span(span)

    # ==================== 查询与调试 ====================

    def get_trace(self, trace_id: str) -> list[Span]:
        """获取指定追踪 ID 的所有跨度。

        Args:
            trace_id: 要查询的追踪 ID

        Returns:
            按时间排序的跨度列表
        """
        # 先从缓存查，再去日志文件查
        spans = [s for s in self._spans if s.trace_id == trace_id]

        if not spans:
            # 从日志文件读取
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if data.get("trace_id") == trace_id:
                                spans.append(Span(**data))
                        except (json.JSONDecodeError, TypeError):
                            continue
            except FileNotFoundError:
                pass

        return spans

    def get_failed_spans(self) -> list[Span]:
        """获取所有失败的跨度。"""
        return [s for s in self._spans if s.error]

    def get_traces_overview(self) -> list[dict]:
        """获取所有追踪的概览（trace_id + 跨度数 + 事件类型列表）。"""
        trace_groups: dict[str, list[Span]] = {}

        for span in self._spans:
            tid = span.trace_id
            if tid not in trace_groups:
                trace_groups[tid] = []
            trace_groups[tid].append(span)

        return [
            {
                "trace_id": tid,
                "span_count": len(spans),
                "event_types": list(set(s.event_type for s in spans)),
                "has_errors": any(s.error for s in spans),
                "first_event": min(s.timestamp for s in spans),
            }
            for tid, spans in trace_groups.items()
        ]

    # ==================== 日志管理 ====================

    def clear_log(self) -> None:
        """清空日志文件（谨慎使用）。"""
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("")
        self._spans.clear()
        self.metrics = Metrics()
        print(f"🗑️ 日志文件已清空：{self.log_file}")

    def get_log_size(self) -> str:
        """获取日志文件大小（人类可读）。"""
        try:
            import os
            size = os.path.getsize(self.log_file)
            if size < 1024:
                return f"{size} B"
            elif size < 1024 * 1024:
                return f"{size / 1024:.1f} KB"
            else:
                return f"{size / 1024 / 1024:.1f} MB"
        except FileNotFoundError:
            return "0 B"

    # ==================== 报告输出 ====================

    def print_summary(self) -> None:
        """打印遥测摘要（指标 + 最新追踪概览）。"""
        m = self.metrics

        print("=" * 50)
        print("  TELEMETRY SUMMARY")
        print("=" * 50)
        print()

        # LLM 调用
        print(f"  LLM Calls:      {m.llm_calls}")
        print(f"    Success Rate:  {m.llm_success_rate * 100:.2f}%")
        print(f"    Avg Latency:   {m.avg_latency_ms:.0f}ms")
        print(f"    Retries:       {m.llm_retries}")

        # 工具调用
        print(f"  Tool Calls:     {m.tool_calls}")
        print(f"    Success Rate:  {m.tool_success_rate * 100:.2f}%")

        # 记忆操作
        print(f"  Memory Ops:     {m.memory_ops}")

        # 日志信息
        log_size = self.get_log_size()
        unique_traces = len(set(s.trace_id for s in self._spans))
        print(f"\n  Log File:       {self.log_file} ({log_size})")
        print(f"  Active Traces:  {unique_traces}")

        print("=" * 50)

    def print_trace_detail(self, trace_id: str) -> None:
        """打印指定追踪的详细信息。

        Args:
            trace_id: 要打印的追踪 ID
        """
        spans = self.get_trace(trace_id)

        if not spans:
            print(f"❌ 未找到追踪：{trace_id}")
            return

        print(f"\n{'=' * 55}")
        print(f"  TRACE: {trace_id}")
        print(f"{'=' * 55}")
        print(f"  Spans: {len(spans)}")
        print()

        for i, span in enumerate(spans, 1):
            event = span.event_type
            dur = f"{span.duration_ms:.0f}ms" if span.duration_ms is not None else "?"

            # 事件图标
            icon = {
                "llm_call": "🤖",
                "tool_request": "🔧",
                "tool_execution": "⚙️",
                "memory_op": "🧠",
                "decision": "🎯",
            }.get(event, "📝")

            status = "❌" if span.error else "✅"

            print(f"  {i}. {icon} {event} {status}")
            print(f"     Span:  {span.span_id}")
            print(f"     Time:  {span.timestamp}")
            print(f"     Dur:   {dur}")

            if span.data:
                data_str = json.dumps(span.data, ensure_ascii=False)
                if len(data_str) > 80:
                    data_str = data_str[:77] + "..."
                print(f"     Data:  {data_str}")

            if span.error:
                print(f"     Error: {span.error}")

            print()


# ==================== 便捷上下文管理器 ====================


class trace:
    """追踪上下文管理器。

    用法：
        with trace(telemetry) as trace_id:
            # 所有在此块内的操作自动关联到该 trace_id
            agent.generate(...)
    """

    def __init__(self, telemetry: Telemetry):
        self.telemetry = telemetry
        self._previous_trace_id = None

    def __enter__(self) -> str:
        self._previous_trace_id = self.telemetry.current_trace_id
        return self.telemetry.start_trace()

    def __exit__(self, *args) -> None:
        self.telemetry.current_trace_id = self._previous_trace_id
