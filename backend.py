import requests
import pandas as pd
from groq import Groq
from collections import Counter
import math
import time

SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
NUTRIENT_MAP = {
    "protein": "Protein",
    "calories": "Energy",
    "fiber": "Fiber, total dietary",
    "sodium": "Sodium, Na",
    "fat": "Total lipid (fat)",
    "carbs": "Carbohydrate, by difference"
}


def get_nutrient_value(food, nutrient_name):
    food_nutrients = food.get("foodNutrients", [])
    for nutrient in food_nutrients:
        if nutrient.get("nutrientName") == nutrient_name:
            return nutrient.get("value", None)
    return None


def search_specific_food(input):
  parameters = {"query": input,"api_key": USDA_API_KEY}
  response = requests.get('https://api.nal.usda.gov/fdc/v1/foods/search',params = parameters)
  response = response.json()
  outputResponse = ""
  data_count = 0
  for food in response.get('foods', []):
        if food.get('dataType') == 'Branded':
            continue

        #print("This food is a",food.get('dataType'))

        #print(food.get('description'))
        #print("Score: ",food.get('score'))
        outputResponse += str(data_count+1) + ")"
        outputResponse += food.get('description') + "\n"
        if food.get('ingredients') is not None:
          outputResponse += "Ingredients: " + food.get('ingredients') +"\n"


        for nutrient in food.get('foodNutrients', []):

            if nutrient.get('nutrientName')in NUTRIENT_MAP.values():
              if food.get('dataType') == 'SR Legacy' and nutrient.get('nutrientName') == 'Energy' and nutrient.get('unitName') == 'kJ':
                    continue
              #print("Name:",nutrient.get('nutrientName'), "  Value: ",nutrient.get('value')," "+nutrient.get('unitName'))
              outputResponse = outputResponse + str(nutrient.get('nutrientName')) + " Value: " + str(nutrient.get('value')) + str(nutrient.get('unitName')) + "\n"
        outputResponse = outputResponse + "\n"
        #print()
        data_count+=1
        if data_count >=3:
          break

  return outputResponse




def get_food(input):
    p={"query":input,"api_key":USDA_API_KEY}
    response =requests.get("https://api.nal.usda.gov/fdc/v1/foods/search", params=p)
    response =response.json()
    return response.get("foods",[])


def rank_foods(food_name,nutrient_key, topn=5, highest=True):
    if nutrient_key not in NUTRIENT_MAP:
        print("Sorry, we do not have this information!.")
        return
    nutrient=NUTRIENT_MAP[nutrient_key]
    foods = get_food(food_name)
    ranked_list =[]
    for food in foods:
        if food.get("dataType") =="Branded":
            continue
        description = food.get("description","Unknown Food")
        data_type = food.get("dataType","Unknown Type")
        nutrient_value = get_nutrient_value(food,nutrient)
        if nutrient_value is not None:
            ranked_list.append({"Food": description,"Type": data_type, nutrient_key.capitalize(): nutrient_value})
    ranked_list = sorted(ranked_list,key=lambda x: x[nutrient_key.capitalize()],reverse=highest)
    df =pd.DataFrame(ranked_list[:topn])
    return df





client = Groq(api_key=GROQ_API_KEY)
###FOR FOOD NAME EXTRACTION

def food_extraction(query, contextOfSession):
   response = client.chat.completions.create(
   messages=[
            {
                "role": "system",
                "content": "You are a food name extractor. Extract and return the exact food item mentioned by the user. If the query gives you a broader food name, return the broad food name. Do not over-specify the food name. If the query has no food name, reply None. Return only the food name or None."
            },
            {
                "role": "user",
                "content": f"User query: {query}"
            }
        ],
        model="llama-3.1-8b-instant",
        temperature=0
    )
   response = response.choices[0].message.content
   return response


client = Groq(api_key=GROQ_API_KEY)

###FOR NORMAL CONVERSATION
def importgroq(usda_info, query, contextOfSession):
    # This prompt forces a hard stop on medical advice
    system_msg = (
        "You are a concise nutrition-assistant chatbot. If the user asks for personalized responses, "
        "medical advice, weight loss plans, or health instructions, respond ONLY with: 'This is not meant "
        "for medical advice. Consult a trained professional.' Stop immediately. Do not provide calorie "
        "deficits or diet instructions. If the query is about food facts, use the provided USDA info."
    )

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"User query: {query}\n USDA Information: {usda_info}\n Chat history: {str(contextOfSession)}"}
        ],
        model="llama-3.1-8b-instant"
    )
    return response.choices[0].message.content




#types of scores
intent_corpus = {
    "lookup": "how many calories in protein in amount of vitamin c in nutrition facts for what is the content value",
    "comparison": "is higher than more protein than compare vs versus healthier difference between better than",
    "ranking": "highest top 5 best most lowest least richest source of rank",
    "educational":"what why when explain advantages benefits purpose outline define"
}

#calculate the scores to categorize the query - match based on total score
def scorecalcjm(query, doc_text, collection_text, lambd=0.5):
    queryTerms =query.lower().split()
    docTerms = doc_text.lower().split()
    collectionTerms = collection_text.lower().split()
    doc_count =Counter(docTerms)
    coll_count = Counter(collectionTerms)
    doc_len = len(docTerms)
    coll_len =len(collectionTerms)

    log_likelihood= 0.0
    vocab_size =len(set(collectionTerms))

    for word in queryTerms:
        # Ensure we are using the correct count variables (doc_count, coll_count)
        pwd = doc_count[word] /doc_len if doc_len > 0 else 0
        pwc = (coll_count[word] + 0.1)/ (coll_len + (0.1 * vocab_size))
        # smoothing for zero
        smooth = (lambd*pwd)+((1 - lambd) * pwc)
        # make sure value greater than z for no crash
        log_likelihood+= math.log(smooth)
    return log_likelihood


def intent_classification(query):
  Ambiguity_threshold = 0.05 # changed from 2.00
  all_text = " ".join(intent_corpus.values())
  scores = {intent: scorecalcjm(query, text, all_text) for intent, text in intent_corpus.items()}
  predicted_intent = max(scores, key=scores.get)

  #print(f"Scores: {scores}")
  #print(f"Predicted Intent: {predicted_intent}")
  sorted_scores = sorted(scores.values(), reverse=True)
  highest_score = sorted_scores[0]
  secondhighest_score = sorted_scores[1]
  Difference = highest_score - secondhighest_score
  if Difference < Ambiguity_threshold:
    predicted_intent = 'groq_api'
  return predicted_intent


def normalize_entity(extracted_text):
    if not extracted_text or extracted_text.lower() == 'none':
        return None
    term = extracted_text.lower().strip().strip("'").strip('"')
    noise_words = ['fresh', 'raw', 'organic', 'frozen', 'ripe', 'a', 'an', 'some']
    words = term.split()
    filtered_words = [w for w in words if w not in noise_words]
    normalized_term = " ".join(filtered_words)
    if normalized_term.endswith('ies'):
        normalized_term = normalized_term[:-3] + 'y'
    elif normalized_term.endswith('s') and not normalized_term.endswith('ss'):
        normalized_term = normalized_term[:-1]
    return normalized_term.strip()

def extract_comparison_foods(query,extracted_food):
    q = query.lower()
    if " vs " in q:
        parts = q.split(" vs ")
    elif " versus " in q:
        parts = q.split(" versus ")
    elif " than " in q:
        parts = q.split(" than ")
    else:
        return [extracted_food]
    food1 = parts[0].replace("compare", "").replace("the fat in", "").replace("is", "").replace("healthier", "").strip()
    food2 = parts[1].replace("?", "").strip()
    return [normalize_entity(food1), normalize_entity(food2)]

def directQuery(query, food, contextOfSession):
    q = query.lower()
    if "lose weight" in q or "weight loss" in q or "what foods should i eat" in q:
        response = importgroq(None, query, contextOfSession)
        return 'groq_api', response

    food = normalize_entity(food)
    intent =intent_classification(query)

    #check when there is no food
    if food is None or food.lower()== 'none':
        response= importgroq(None, query, contextOfSession)
        return 'groq_api', response

    if any(word in query.lower() for word in ['recipe', 'diet', 'plan', 'meal']):
        intent = 'groq_api'
        response = importgroq(None, query, contextOfSession)
        return intent, response

    #return info from usda
    if intent=='lookup':
        response =search_specific_food(food)
    # get data from foods to compare
    elif intent== 'comparison':
        # Logic: Fetch data for the food(s) and let Groq explain the differences
        foods_to_compare = extract_comparison_foods(query, food)
        relevant_info = ""
        for item in foods_to_compare:
            relevant_info += f"\nNutrition information for {item}:\n"
            relevant_info += search_specific_food(item)
        response = importgroq(relevant_info, query, contextOfSession)        
        intent = 'comparison'
        # use groq_api so response is treated as text

    #use nutrition key for rank then default to cals if not availible
    elif intent=='ranking':
        nutrient_key= "calories"
        for key in NUTRIENT_MAP.keys():
            if key in query.lower():
                nutrient_key= key
        # return a framing of the rank foods
        response = rank_foods(food, nutrient_key, topn=5, highest=True)

    #use general info from groq
    elif intent== 'educational':
        relevant_info= search_specific_food(food)
        response= importgroq(relevant_info, query, contextOfSession)

    elif intent == 'groq_api':
        relevant_info = search_specific_food(food)
        response =importgroq(relevant_info, query, contextOfSession)
    
    return intent, response



# Initialize contextOfSession
contextOfSession = []


def run_system_evaluation():

    # Definine the eval dataset
    # Format: (Input Query, Expected Intent, Expected Food Entity)
    test_cases = [
        # Lookup Tests
        ("How many calories are in a red apple?", "lookup", "apple"),
        ("What is the protein content of an egg?", "lookup", "egg"),

        # Ranking Tests
        ("What are the top 5 foods highest in fiber?", "ranking", "food"),
        ("Which pasta has the most protein?", "ranking", "pasta"),

        # Comparison Tests (Currently redirected to Groq/Educational)
        ("Compare the fat in salmon vs chicken breast", "comparison", "salmon, chicken breast"),
        ("Is kale healthier than spinach?", "comparison", "kale, spinach"),

        # Guardrail/LLM Fallback Tests
        ("I want to lose 20 pounds, give me a meal plan", "groq_api", None),
        ("What makes blueberries a superfood?", "groq_api", "blueberries"),
        ("Is this app better than a doctor?", "groq_api", None),
        ("Give me a recipe for keto brownies", "groq_api", "brownies")
    ]

    results = []
    intent_matches = 0
    food_matches = 0

    print("-" * 50)
    print("NutriPal Evaluation Runner")
    print(f"Total Test Cases: {len(test_cases)}")
    print("-" * 50)

    for query, expected_intent, expected_food in test_cases:
        print(f"Processing: '{query}'...")
        time.sleep(2) # for API rate limits

        try:
            # Step A: Run Extraction (uses defined food_extraction)
            extracted_food = food_extraction(query, [])

            # Step B: Run the Logic (uses defined directQuery)
            predicted_intent, response = directQuery(query, extracted_food, [])

            # Step C: Scoring Logic
            # Updated Scoring Logic for Final Submission
            is_intent_correct = (predicted_intent == expected_intent)
            is_food_correct = False

            if expected_food is None:
                is_food_correct = (extracted_food is None or extracted_food == "None")
            elif expected_food.lower() == "food":
                # If we expect 'food', any extracted food item is a success
                is_food_correct = (extracted_food is not None and extracted_food != "None")
            else:
                # Check for overlap between expected and extracted
                is_food_correct = (expected_food.lower() in extracted_food.lower() or
                                   extracted_food.lower() in expected_food.lower())

            if is_intent_correct: intent_matches += 1
            if is_food_correct: food_matches += 1

            results.append({
                "User Query": query,
                "Expected Intent": expected_intent,
                "Predicted Intent": predicted_intent,
                "Intent Match": "PASS" if is_intent_correct else "FAIL",
                "Extracted Food": extracted_food,
                "Food Match": "PASS" if is_food_correct else "FAIL"
            })

            # needed for API rate limits
            time.sleep(0.5)

        except Exception as e:
            print(f"Error on query '{query}': {e}")
            continue

    # 2. Calculation of Metrics
    total = len(test_cases)
    intent_accuracy = (intent_matches / total) * 100
    food_accuracy = (food_matches / total) * 100

    # 3. Displaying Results
    df = pd.DataFrame(results)

    print("\n" + "="*30)
    print("EVALUATION SUMMARY")
    print("="*30)
    print(f"Intent Classification Accuracy: {intent_accuracy:.2f}%")
    print(f"Entity Extraction Accuracy:    {food_accuracy:.2f}%")

    # Define Threshold
    # 80% success threshold
    SYSTEM_THRESHOLD = 80.0
    if intent_accuracy >= SYSTEM_THRESHOLD:
        print(f"\nSTATUS: SUCCESS (System exceeded the {SYSTEM_THRESHOLD}% threshold)")
    else:
        print(f"\nSTATUS: NEEDS OPTIMIZATION (System below {SYSTEM_THRESHOLD}% threshold)")

    return df
