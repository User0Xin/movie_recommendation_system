<template>
  <div class="container">
    <div class="card">
      <!-- 用户ID输入框 -->
      <div class="form-group">
        <label>用户ID：</label>
        <input v-model="uid" type="text" placeholder="请输入用户ID" class="input" />
      </div>

      <!-- 主体布局 -->
      <div class="layout">

        <!-- 聊天框 -->
        <div class="chat-box">
          <div class="chat-messages" ref="scrollArea">
            <div v-for="(msg, i) in messages" :key="i" :class="msg.role === 'user' ? 'msg user' : 'msg bot'">
              <div class="bubble" v-html="msg.text"></div>
            </div>

            <div v-if="loading" class="loading-dots">
              <span></span><span></span><span></span>
            </div>
          </div>

          <div class="chat-input">
            <input v-model="inputMessage" @keyup.enter="sendMessage" placeholder="输入你的问题..." class="input" />
            <button @click="sendMessage" class="btn chat-btn">发送</button>
          </div>
        </div>

        <!-- 推荐栏 -->
        <div class="rec-box">
          <div class="rec-header">
            <h2>电影推荐</h2>
            <button @click="fetchRecommendations" class="btn green">获取TOP10</button>
          </div>

          <div class="rec-list">
            <div v-if="recLoading" class="spinner"></div>

            <ul v-else>
              <li v-for="(movie, index) in recommendations" :key="index" class="rec-item">
                {{ index + 1 }}. {{ movie }}
              </li>
            </ul>
          </div>
        </div>

      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick } from "vue";
import { recommendMoviesApi, agentQuery } from '@/api/movieRecommendApi'
import { marked } from "marked";
const uid = ref("");
const messages = ref([]);
const inputMessage = ref("");
const loading = ref(false);
const recLoading = ref(false);
const recommendations = ref([]);
const scrollArea = ref(null);

async function sendMessage() {
  if (!inputMessage.value || !uid.value) return;

  messages.value.push({ role: "user", text: inputMessage.value });
  const userInput = inputMessage.value;
  inputMessage.value = "";
  loading.value = true;

  try {
    const response = await agentQuery(uid.value, userInput);
    const html = marked.parse(response);
    messages.value.push({ role: "assistant", text: html });
  } catch (e) {
    messages.value.push({ role: "assistant", text: "服务器错误，请稍后再试。" });
  }

  loading.value = false;

  nextTick(() => {
    if (scrollArea.value) scrollArea.value.scrollTop = scrollArea.value.scrollHeight;
  });
}

async function fetchRecommendations() {
  if (!uid.value) return alert("请先输入用户ID！");

  recLoading.value = true;
  recommendMoviesApi(uid.value).then((response) => {
    recommendations.value = response;
    recLoading.value = false;
  }).catch(() => {
    recommendations.value = ["获取失败，请稍后再试"];
    recLoading.value = false;
  });
}
</script>

<style scoped>
ul {
  list-style-type: none;
  padding: 0;
  margin: 0;
}
/* 基础布局 */
.container {
  min-height: 100vh;
  width: 1280px;
  margin: 0 auto;
  background: #f2f2f2;
  padding: 30px;
  display: flex;
  justify-content: center;
}

.card {
  background: white;
  width: 100%;
  max-width: 1200px;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.form-group {
  margin-bottom: 20px;
}

.label { font-weight: bold; }

.input {
  width: 100%;
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 8px;
  font-size: 14px;
}

/* 主体三列布局 */
.layout {
  display: grid;
  grid-template-columns: 2fr 1fr;
  height: 85%;
  gap: 20px;
}

/* 聊天框 */
.chat-box {
  background: #fafafa;
  border-radius: 12px;
  border: 1px solid #ddd;
  display: flex;
  flex-direction: column;
  height: 100%;
  padding: 10px;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding-right: 5px;
}

.chat-btn{
  width: 70px;
}

.msg {
  margin: 10px 0;
  display: flex;
}

.msg.user { justify-content: flex-end; }
.msg.bot { justify-content: flex-start; }

.bubble {
  max-width: 70%;
  padding: 10px 12px;
  border-radius: 10px;
  background: #e0e0e0;
}

.msg.user .bubble {
  background: #4a90e2;
  color: white;
}

/*等待动画*/
.loading-dots span {
  width: 8px;
  height: 8px;
  background: #aaa;
  border-radius: 50%;
  display: inline-block;
  margin: 0 3px;
  animation: blink 1.4s infinite both;
}

.loading-dots span:nth-child(2) { animation-delay: 0.2s; }
.loading-dots span:nth-child(3) { animation-delay: 0.4s; }

@keyframes blink {
  0%, 80%, 100% { opacity: 0; }
  40% { opacity: 1; }
}

.chat-input {
  display: flex;
  gap: 10px;
  margin-top: 10px;
}

.btn {
  padding: 10px 15px;
  background: #4a90e2;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
}

.btn.green { background: #28a745; }

/* 推荐栏 */
.rec-box {
  background: #fafafa;
  border-radius: 12px;
  border: 1px solid #ddd;
  padding: 10px;
  height: 100%;
  display: flex;
  flex-direction: column;
}

.rec-header {
  margin-top: 5px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
  padding: 0 2px;
}

.rec-list {
  flex: 1;
  overflow-y: auto;
}

.rec-item {
  background: white;
  padding: 10px;
  margin-bottom: 8px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

/* 加载 spinner */
.spinner {
  width: 30px;
  height: 30px;
  border: 4px solid #ccc;
  border-top-color: #28a745;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin: 20px auto;
}

@keyframes spin {
  100% { transform: rotate(360deg); }
}
</style>
