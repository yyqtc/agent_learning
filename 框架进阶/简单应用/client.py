from langserve import RemoteRunnable

remote_chain = RemoteRunnable(url="http://localhost:8000/chain/")
result = remote_chain.invoke({
    "language": "italian",
    "text": "橘子"
})

print(result)