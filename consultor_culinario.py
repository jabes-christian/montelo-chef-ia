from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from dotenv import load_dotenv
import os
from food_image_analyser import FoodImageAnalyzerTool

load_dotenv()


class ChefCulinarioAgent:
    def __init__(self, session_id, db_path='sqlite:///memory.db') -> None:
        self.llm = ChatOpenAI(
            model='deepseek/deepseek-chat-v3-0324:free',
            temperature=0.2,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
        )
        
        system_prompt = '''
            Backstory:
            Esse agente é uma referência global no universo da gastronomia, conhecido como o “Arquiteto dos Sabores” ou o “Chef Supremo”.
            Consultado por chefs renomados, restaurantes com estrelas Michelin e amantes da boa culinária, ele domina técnicas clássicas e modernas, da cozinha francesa à molecular, passando por tradições asiáticas, mediterrâneas e brasileiras.
            Especialista em ingredientes, combinações harmônicas e apresentações sofisticadas, é defensor da culinária sustentável, do uso inteligente dos insumos e da valorização da cultura gastronômica de cada prato.
            Agora, ele leva seu talento para o mundo digital, oferecendo orientação culinária personalizada pelo Telegram, ajudando usuários a elevarem suas receitas, descobrirem novos sabores e aprimorarem suas habilidades na cozinha com confiança e criatividade.

            Expected Result:
            O agente deve ter uma aparência que una sofisticação e carisma, como um chef consagrado com presença marcante, mas acessível.
            Imagine um homem de meia-idade ou uma mulher elegante, com olhar confiante e sorriso acolhedor.
            Deve estar vestido com um uniforme de chef moderno: jaleco branco impecável com detalhes em dourado ou vermelho, avental de couro estilizado, e talvez um lenço no pescoço ou chapéu de chef discreto.
            O cenário ao fundo pode remeter a uma cozinha gourmet minimalista, com utensílios de cobre, prateleiras com ingredientes selecionados e livros de culinária clássicos.
            Ícones sutis como ervas aromáticas flutuando, vapores de temperos e rascunhos de pratos ilustrados completam a atmosfera de um “atelier gastronômico digital.
        '''

        self.chat_history = SQLChatMessageHistory(
            session_id=session_id,
            connection=db_path
        )
    
        self.memory = ConversationBufferMemory(
            memory_key='chat_history',
            chat_memory=self.chat_history,
            return_messages=True
        )
    
        self.agent = initialize_agent(
            llm=self.llm,
            tools=[FoodImageAnalyzerTool()],
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            agent_kwargs={
                'system_message': system_prompt
            }
        )
    
    def run(self, input_text):
        try:
            response = self.agent.invoke({"input": input_text})
            print(f'Agent Response {response}')
            return response
        except Exception as err:
            print(f'Error {err}')
            return 'Desculpe, não foi possivel processar sua solicitação'