"""Control routes stay separate from executable business intent labels."""

ORCHESTRATE_ROUTE = "orchestrate"
ORCHESTRATE_DESCRIPTION = (
    "同一消息要求分别回应或处理多个独立诉求，可能跨业务领域或存在先后依赖。"
    "例如先排查登录问题，再解释退款条件；查询套餐，同时询问发票规则。"
    "原因、背景、付款事实、套餐参数、已完成事项、否定事项不构成新增诉求。"
    "同一业务问题的多个参数或细节不需要拆解。"
)
