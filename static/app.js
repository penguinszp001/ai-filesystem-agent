document.addEventListener("DOMContentLoaded", () => {

    const chatDiv = document.getElementById("chat");
    const input = document.getElementById("input");
    const status = document.getElementById("status");
    const sendBtn = document.getElementById("sendBtn");

    async function init() {
        try {
            const res = await fetch("/info");
            const data = await res.json();

            document.getElementById("directory").innerText =
                "📁 Directory: " + data.directory;
        } catch (err) {
            console.error("Failed to load directory:", err);
        }
    }

    function addMessage(text, role) {
        const div = document.createElement("div");
        div.className = "message " + role;
        div.innerText = text;
        chatDiv.appendChild(div);
        chatDiv.scrollTop = chatDiv.scrollHeight;
    }

    async function send() {
        const text = input.value.trim();
        if (!text) return;

        addMessage(text, "user");
        input.value = "";

        status.innerText = "Thinking...";

        try {
            const res = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: text })
            });

            const data = await res.json();

            addMessage(data.response, "assistant");
            status.innerText = "Done";

        } catch (err) {
            console.error(err);
            status.innerText = "Error";
        }
    }

    // 🔥 button click
    sendBtn.addEventListener("click", send);

    // 🔥 ENTER key
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            send();
        }
    });

    init();
});