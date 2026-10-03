import { BrainCircuit, CheckCircle2, ChevronDown, Database, Puzzle, Route } from 'lucide-react'
import { useMemo, useState } from 'react'
import { supervisorIntentLabels, supervisorRewrite } from '../api/tokenplan'
import type { ChatResponse, IntentExecution } from '../api/tokenplan'
import { labelAgent, labelIntent } from '../utils/labels'

const stageLabels: Record<string, string> = {
  few_shot_retrieval_ms: 'Few-shot 检索',
  supervisor_ms: 'Supervisor',
  binding_ms: '能力绑定',
  intent_queue_wait_ms: '委派排队',
  agent_decision_ms: 'Agent 决策',
  tool_execution_ms: '工具执行',
  worker_execution_ms: '领域执行',
  response_guard_ms: '回答校验',
  total_ms: '总耗时',
}

function unique(values: Array<string | undefined>): string[] {
  return [...new Set(values.filter((value): value is string => Boolean(value)))]
}

function skillsOf(executions: IntentExecution[]): string[] {
  return unique(executions.flatMap((item) => item.selected_skill_ids ?? []))
}

function compactDuration(value: number): string {
  return value >= 1000 ? `${(value / 1000).toFixed(1)}s` : `${Math.round(value)}ms`
}

export function ExecutionPanel({ result }: { result: ChatResponse }) {
  const [open, setOpen] = useState(false)
  const skills = useMemo(() => skillsOf(result.intent_executions ?? []), [result.intent_executions])
  const executions = result.intent_executions ?? []
  const intents = supervisorIntentLabels(result)
  const rewrite = supervisorRewrite(result)
  const stageTimings = Object.entries(result.stage_timings_ms ?? {}).filter(([, value]) => Number.isFinite(value))
  const rewritten = rewrite.status === 'resolved' && typeof rewrite.effective_query === 'string'

  return (
    <div className="execution-panel">
      <button type="button" className="execution-toggle" onClick={() => setOpen((value) => !value)}>
        <span>
          <BrainCircuit aria-hidden="true" />
          查看本次处理过程
        </span>
        <span className="execution-summary">
          {intents.length} 个意图 · {result.agent_types.length || (result.agent_type ? 1 : 0)} 个 Agent · {compactDuration(result.latency_ms)}
          <ChevronDown className={open ? 'rotate' : ''} aria-hidden="true" />
        </span>
      </button>

      {open ? (
        <div className="execution-body">
          <section className="execution-section">
            <div className="execution-section-title"><Route aria-hidden="true" />意图与委派</div>
            <div className="chip-row">
              {intents.map((intent) => <span className="meta-chip intent" key={intent}>{labelIntent(intent)}</span>)}
              {unique(result.agent_types.length ? result.agent_types : [result.agent_type]).map((agent) => (
                <span className="meta-chip agent" key={agent}>{labelAgent(agent)}</span>
              ))}
            </div>
            {executions.length ? (
              <div className="execution-list">
                {executions.map((execution, index) => (
                  <div className="execution-item" key={execution.intent_id ?? `${execution.intent}-${index}`}>
                    <span className={`execution-dot ${execution.status === 'COMPLETED' ? 'success' : ''}`} />
                    <span className="execution-item-copy">
                      <strong>{labelIntent(String(execution.intent ?? '未知意图'))}</strong>
                      <small>
                        {labelAgent(String(execution.agent_type ?? ''))}
                        {typeof execution.latency_ms === 'number' ? ` · ${compactDuration(execution.latency_ms)}` : ''}
                      </small>
                    </span>
                    <span className="execution-status">{execution.status ?? 'UNKNOWN'}</span>
                  </div>
                ))}
              </div>
            ) : null}
          </section>

          <section className="execution-section">
            <div className="execution-section-title"><Puzzle aria-hidden="true" />Skill 与知识</div>
            <div className="fact-grid">
              <div>
                <span>已加载 Skill</span>
                <strong>{skills.length ? skills.join('、') : '本次未加载'}</strong>
              </div>
              <div>
                <span>知识检索</span>
                <strong>{result.knowledge_used ? `已使用 · ${result.evidence_ids.length} 条证据` : '本次未使用'}</strong>
              </div>
              <div>
                <span>会话记忆</span>
                <strong>{result.memory_persisted ? '已更新' : '回答成功，记忆写入失败'}</strong>
              </div>
            </div>
          </section>

          {rewritten || stageTimings.length ? (
            <section className="execution-section">
              <div className="execution-section-title"><Database aria-hidden="true" />Query 与耗时</div>
              {rewritten ? (
                <div className="rewrite-box">
                  <span>Supervisor 补全后</span><p>{String(rewrite.effective_query)}</p>
                </div>
              ) : null}
              {stageTimings.length ? (
                <div className="timing-list">
                  {stageTimings.map(([name, value]) => (
                    <span key={name}><small>{stageLabels[name] ?? name}</small><strong>{compactDuration(value)}</strong></span>
                  ))}
                </div>
              ) : null}
            </section>
          ) : null}

          <div className="trace-footer">
            <CheckCircle2 aria-hidden="true" />
            <span>{result.escalated ? '已转人工处理' : 'Supervisor 已完成统一汇总'}</span>
            {result.trace_id ? <code>{result.trace_id}</code> : null}
          </div>
        </div>
      ) : null}
    </div>
  )
}
