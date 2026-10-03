# TokenPlan Frontend

面向 TokenPlan 订阅客服智能体的演示工作台，展示多意图识别、Supervisor 委派、领域 Agent、Skill、知识检索和会话记忆。处理结果以正常对话形式呈现，意图和 Agent 以简洁标签标注；需要排查时，可以展开回答下方的处理过程。

## 本地启动

先在项目根目录启动后端，使 `http://127.0.0.1:8000/health` 可访问，然后执行：

```powershell
cd frontend
npm install
npm run dev
```

浏览器访问 `http://127.0.0.1:5173`。开发服务器默认将 `/api` 代理到 `http://127.0.0.1:8000`；后端地址不同时，可在启动前设置 `TOKENPLAN_BACKEND_URL`。

如需让构建产物直接请求一个完整 API 地址，可配置：

```powershell
$env:VITE_TOKENPLAN_API_BASE_URL = 'https://example.com'
npm run build
```

## 验证

```powershell
npm test
npm run lint
npm run build
```
