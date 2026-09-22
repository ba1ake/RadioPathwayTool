import discord, os
from dotenv import load_dotenv
load_dotenv()


intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event # starts the conneciton from server to discord
async def on_ready():
    print(f'We have logged in as {client.user}') # prints the bot name into the console

@client.event # when a messgae comes in
async def on_message(message): 
    if message.author == client.user: #message.auther = itself
        return

    if message.content.startswith('$hello'): # mesasge content object starts with x
        await message.channel.send('Hello!') # send a response to the channel in the message object

client.run(os.getenv("DISCORD_BOT_TOKEN"))