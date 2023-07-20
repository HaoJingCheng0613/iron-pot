import asyncio
import websockets
import IF_choose

# 定义全局变量保存服务器对象
server = None

async def echo(websocket, path):
    i=0
    async for message in websocket:
        if message == "clear":
            pass
        
        else:    
            print(message)
            i=i+1
            print(i)
            choose = IF_choose.if_choose(message)
            if not websocket.closed:  # 检查连接是否已经关闭
                await websocket.send(choose)
                pass

async def main():
    global server
    server = await websockets.serve(echo, "localhost", 2333)  # 保存服务器对象
    try:
        await server.wait_closed()  # 等待服务器关闭
    except asyncio.CancelledError:
        server.close()  # 在服务器关闭时触发异常
        await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:  # 捕获键盘中断
        print("服务停止")
