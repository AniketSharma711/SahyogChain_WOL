import os
import shutil
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
import docx.oxml.simpletypes as st

# Patch Twips conversion bug for non-integer twip values in existing docx
orig_twips_convert = st.ST_TwipsMeasure.convert_from_xml
def patched_twips_convert(cls, str_value: str):
    if "i" in str_value or "m" in str_value or "p" in str_value:
        return st.ST_UniversalMeasure.convert_from_xml(str_value)
    try:
        return st.Twips(int(str_value))
    except ValueError:
        return st.Twips(int(round(float(str_value))))

st.ST_TwipsMeasure.convert_from_xml = classmethod(patched_twips_convert)


def set_p_text(p, text, font_name="Times New Roman", font_size=12, bold=False, italic=False, align=None):
    p.text = ""
    run = p.add_run(text)
    run.font.name = font_name
    run.font.size = Pt(font_size)
    run.bold = bold
    run.italic = italic
    if align is not None:
        p.alignment = align
    return run


def set_p_lead_in(p, lead_in, body, font_name="Times New Roman", font_size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p.text = ""
    r1 = p.add_run(lead_in)
    r1.font.name = font_name
    r1.font.size = Pt(font_size)
    r1.bold = True
    
    r2 = p.add_run(" " + body)
    r2.font.name = font_name
    r2.font.size = Pt(font_size)
    r2.bold = False
    p.alignment = align
    return r1, r2


def format_cell(cell, text, bold=False, font_size=10, align=WD_ALIGN_PARAGRAPH.LEFT):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(font_size)
    run.bold = bold
    return run


def update_table(table, data):
    while len(table.rows) < len(data):
        table.add_row()
    for row_idx, row in enumerate(data):
        for col_idx, text in enumerate(row):
            format_cell(table.cell(row_idx, col_idx), text, bold=(row_idx == 0), font_size=10)


def embed_fig(p, img_path, caption_text, width_inches=5.2):
    p.text = ""
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(img_path, width=Inches(width_inches))
    p.add_run("\n")
    r_cap = p.add_run(caption_text)
    r_cap.font.name = "Times New Roman"
    r_cap.font.size = Pt(10.5)
    r_cap.bold = True
    r_cap.italic = True


def main():
    doc_path = "Summer_Training_Report_backup.docx"
    print(f"Loading {doc_path}...")
    doc = docx.Document(doc_path)
    paras = doc.paragraphs
    tables = doc.tables

    # -------------------------------------------------------------
    # 1. COVER PAGE & FRONT MATTER (Cleanly updated for Aniket Sharma)
    # -------------------------------------------------------------
    print("Updating Title Page...")
    set_p_text(paras[7], "Summer Training Report", font_size=24, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    set_p_text(paras[8], "on", font_size=24, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    set_p_text(paras[9], "Autonomous Supply Chain Optimization Using Deep Reinforcement Learning for Rural Women Producer Cooperatives", 
               font_size=18, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    
    set_p_text(paras[12], "Submitted in partial fulfillment of the requirements for the completion of summer internship/training [ARP 355/ART 457]", 
               font_size=14, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER)
    
    set_p_text(paras[14], "Name: Aniket Sharma", font_size=14, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[15], "Enroll No: 01319051923", font_size=14, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[16], "B.Tech AIDS", font_size=14, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT)
    
    set_p_text(paras[18], "Under the supervision of", font_size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    set_p_text(paras[21], "Mr. Santosh Kumar Sharma", font_size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)

    # -------------------------------------------------------------
    # 2. CERTIFICATE & DECLARATION
    # -------------------------------------------------------------
    print("Updating Certificate and Declaration...")
    set_p_text(paras[26], "CERTIFICATE", font_size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    set_p_text(paras[29], "DECLARATION", font_size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    set_p_text(paras[31], 
               "I hereby declare that the Summer Training Report entitled “AUTONOMOUS SUPPLY CHAIN OPTIMIZATION USING DEEP REINFORCEMENT LEARNING FOR RURAL WOMEN PRODUCER COOPERATIVES” is an authentic record of work completed as requirements of Summer Training (ARP 355/ART 457) during the period from 1st July 2025 to 1st August 2025 in Winter of Life NGO under the supervision of Mr. Santosh Kumar Sharma.",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_text(paras[40], "Date: 1st August 2025", font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[41], "Aniket Sharma", font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[53], "Santosh Kumar Sharma", font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT)

    # -------------------------------------------------------------
    # 3. ACKNOWLEDGEMENT (Indian Student Tone)
    # -------------------------------------------------------------
    print("Updating Acknowledgement...")
    set_p_text(paras[55], "ACKNOWLEDGEMENT", font_size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    set_p_text(paras[57], 
               "I would like to express my sincere and heartfelt gratitude to my project mentors, fellow interns, and the entire team at Winter of Life NGO who supported me throughout this summer training. Their constant encouragement, practical ground-level insights, and patient guidance helped me understand how real-world rural supply chains operate and how artificial intelligence can be practically applied to solve everyday problems faced by rural communities.",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_text(paras[58], 
               "Working alongside team members who are genuinely passionate about women empowerment in rural Jharkhand was very inspiring for me. Interacting with field coordinators who work directly with tribal Mahila Samitis (Self-Help Groups) gave me a clear picture of ground realities—such as how milk spoils when roads get flooded in monsoons, or how transport costs jump when private tempos are hired in an emergency. These discussions were critical in helping me design realistic simulation scenarios rather than just relying on textbook assumptions.",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_text(paras[59], 
               "I am especially grateful to Mr. Santosh Kumar Sharma, Vice President of Winter of Life NGO, for giving me the opportunity to take up this meaningful project and for his constant mentorship. His practical suggestions on cooperative economics, understanding the livelihood struggles of rural women producers, and his encouragement to build a working prototype helped me bridge my academic knowledge of Reinforcement Learning with a real social cause. This internship has been a great learning experience for me both technically and personally.",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    # -------------------------------------------------------------
    # 4. ABOUT WINTER OF LIFE NGO (Indian Rural Focus)
    # -------------------------------------------------------------
    print("Updating About NGO...")
    set_p_text(paras[60], "About Winter of Life NGO", font_size=14, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[61], 
               "Winter of Life NGO is a non-profit organization established in New Delhi, working actively towards rural development, grassroots women empowerment, and sustainable community livelihoods. While the central office in Delhi coordinates overall project strategy, institutional linkages, and urban retail partnerships, the NGO’s core fieldwork takes place in remote, tribal, and economically vulnerable pockets of Jharkhand and Uttarakhand.",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_text(paras[62], 
               "In Jharkhand, Winter of Life NGO works closely with tribal women across Khunti, Ranchi, and Hazaribagh districts. The organization helps form and support Swayam Sahayata Samuh (Self-Help Groups or SHGs) and Village Producer Collectives. These women groups produce traditional and local commodities such as fresh desi cow and goat milk, processed ragi (mandua) flour, wild lac (lah), organic forest honey, and handloom crafts like Tussar silk sarees. By aggregating these products through Village Collection Centres, the NGO helps the village didis avoid selling to local middlemen (beparis) at unfair throwaway rates.",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_text(paras[63], 
               "A key priority of Winter of Life NGO is ensuring that rural enterprises run sustainably and become financially independent. The NGO sets up Common Facility Centres (CFCs) where raw produce is cleaned, processed, and packed by trained women operators. Field staff assist with basic accounting, quality checking, and coordinating local transport so that goods can reach district mandis and urban retail outlets on time.",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_text(paras[64], 
               "However, running supply chains in rural Jharkhand comes with serious everyday hurdles—unpaved roads getting washed out during heavy rains, erratic electricity at processing centres, and sudden demand spikes during festivals like Chhath Puja, Sarhul, and Diwali. When logistics fail, perishable produce rots and transport costs shoot up, directly hurting the cooperative's earnings. To tackle this, Winter of Life NGO encouraged the development of this data-driven, AI-based decision support system to help local cooperative supervisors plan inventory, avoid stockouts, and protect the hard-earned profits of rural women.",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    # -------------------------------------------------------------
    # 5. TABLE OF CONTENTS (Table 0)
    # -------------------------------------------------------------
    print("Updating Table of Contents...")
    toc_data = [
        ("S. No.", "Topic", "Page No."),
        ("1", "Abstract", "v"),
        ("2", "Introduction\n1. Rural Cooperative Context and Challenges\n2. Project Objective and Scope", "1"),
        ("3", "Problem Statement & Rural Supply Chain Dynamics\n1. Logistics Hurdles in Rural Jharkhand SHGs\n2. Why Traditional Reorder Rules Fail Under Disruptions\n3. Direct Impact on Tribal Women Producers", "3"),
        ("4", "Description of Training Modules & Work Done\nModule 1: Field Assessment & Rural Supply Chain Mapping\nModule 2: Building the Gymnasium Simulator (SupplyChain-v0)\nModule 3: Training RL Agents (PPO & A2C) with Stable-Baselines3\nModule 4: Designing Heuristic Baselines and Benchmark Tests\nModule 5: Developing the Interactive Streamlit Management Dashboard", "5"),
        ("5", "Methodology & Technical Implementation\n1. Day-to-Day Simulation Flow\n2. Hardware, Software & Libraries Used\n3. Mathematical Formulation (MDP) & Reward Function\n4. Solving Windows Memory Bottlenecks & Practical Tweaks", "7"),
        ("6", "Results and Discussions\n1. Performance Comparison Across 7 Realistic Scenarios\n2. What Did the PPO Agent Actually Learn?\n3. Practical and Socio-Economic Gains for Winter of Life NGO", "11"),
        ("7", "Key Challenges, Learnings & Conclusions\n1. Practical Engineering Challenges Encountered\n2. Personal & Technical Learnings", "15"),
        ("8", "References", "16")
    ]
    update_table(tables[0], toc_data)

    # -------------------------------------------------------------
    # 6. LIST OF FIGURES (Table 1)
    # -------------------------------------------------------------
    print("Updating List of Figures...")
    lof_data = [
        ("Figure No.", "Title", "Page No."),
        ("1.1", "Multi-Echelon Rural Cooperative Network (CFC Hub -> District Hubs -> Customers)", "2"),
        ("1.2", "Daily Step Execution Flow in the SupplyChain-v0 Simulation Environment", "6"),
        ("1.3", "PPO Neural Network Training Progress over 30 Lakh Steps (Convergence Curve)", "8"),
        ("1.4", "Project Codebase and Directory Layout", "9"),
        ("1.5", "Interactive Streamlit Management Dashboard for Rural Cluster Managers", "10"),
        ("1.6", "Cost Breakdown Comparison: Traditional Heuristic vs. PPO (Normal Operations)", "12"),
        ("1.7", "Inventory Level Trajectory During Peak Festive Rush (Stockout Prevention)", "13"),
        ("1.8", "Normalized Performance Scores Across All 7 Disruption Scenarios", "14")
    ]
    update_table(tables[1], lof_data)

    # -------------------------------------------------------------
    # 7. ABBREVIATIONS AND NOMENCLATURE
    # -------------------------------------------------------------
    print("Updating Abbreviations...")
    set_p_text(paras[71], "ABBREVIATIONS AND NOMENCLATURE", font_size=14, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    abbrs = [
        ("WOL", "Winter of Life NGO"),
        ("SHG", "Self-Help Group (Mahila Swayam Sahayata Samuh)"),
        ("CFC", "Common Facility Centre (Village Processing Unit)"),
        ("VLCC", "Village Level Collection Centre"),
        ("CLF", "Cluster Level Federation"),
        ("RL", "Reinforcement Learning"),
        ("DRL", "Deep Reinforcement Learning"),
        ("PPO", "Proximal Policy Optimization"),
        ("A2C", "Advantage Actor-Critic"),
        ("MDP", "Markov Decision Process"),
        ("GAE", "Generalized Advantage Estimation"),
        ("FEFO", "First-Expired, First-Out (Inventory Shelf-Life Rule)"),
        ("FIFO", "First-In, First-Out (Processing Queue Rule)"),
        ("JIT", "Just-in-Time (Lean Inventory Ordering)"),
        ("FTL", "Full Truckload Transport (Full Load Mini-Truck)"),
        ("LTL", "Less Than Truckload (Part Load with Extra Surcharge)"),
        ("COGS", "Cost of Goods Sold (Procurement Cost of Units Sold)"),
        ("OTIF", "On-Time In-Full Delivery Rate"),
        ("DSS", "Decision Support System (Management Dashboard)")
    ]
    for idx, (abbr, full) in enumerate(abbrs):
        target_p = paras[73 + idx]
        set_p_text(target_p, f"{abbr}\t\t{full}", font_size=11, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT)

    # -------------------------------------------------------------
    # 8. ABSTRACT (Indian Rural Context, No USD)
    # -------------------------------------------------------------
    print("Updating Abstract...")
    set_p_text(paras[91], "ABSTRACT", font_size=14, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[92], 
               "This report presents the work done during my one-month summer internship at Winter of Life NGO from 1st July to 1st August 2025. The project focused on solving a very real logistical problem: optimizing multi-tier supply chain and inventory decisions for rural women producer cooperatives in Jharkhand using Deep Reinforcement Learning. In districts like Khunti, Ranchi, and Hazaribagh, women self-help groups (SHGs) collect and process perishable dairy, traditional millets (ragi/mandua), and tribal handicrafts. However, because of frequent disruptions like monsoon road washouts, processing unit power cuts, and sudden demand spikes during festivals like Chhath Puja, these small cooperatives face severe milk spoilage, transport penalties, and lost sales.",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_text(paras[93], 
               "To tackle this, I developed a custom Gymnasium-based simulation environment called SupplyChain-v0. The environment simulates a three-tier network: an upstream processing centre (CFC), three regional storage hubs, and three consumer markets. It models daily operations over a 60-day horizon with real-world constraints—such as batch shelf life, truck capacity (160 units), extra charges for part-loads (LTL freight), and overtime processing costs. The problem was formulated as a Markov Decision Process (MDP) with a 94-dimensional continuous observation vector and a 19-dimensional MultiDiscrete action space. To make sure the agent optimizes real business profit rather than exploiting short-term quirks, I implemented an accrual accounting reward function where goods are charged only when sold or wasted.",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_text(paras[94], 
               "I trained a Proximal Policy Optimization (PPO) agent from Stable-Baselines3 across six randomized disruption scenarios for 30 Lakh steps and benchmarked it against five hand-written heuristics across 210 test episodes. The results showed that PPO achieved a normalized score of 116.2 (where uniform random is 0 and the best forecast-based heuristic is 100). On average, PPO earned ₹10,700 more net profit per 60-day cycle (earning ₹79,500 vs. ₹70,100 in normal operations, and ₹91,300 vs. ₹76,000 in peak festive seasons). Crucially, PPO learned 'production smoothing'—it reduced expensive emergency overtime shifts from 74% to 42% of days and kept a sensible buffer stock before festive peaks. In an unseen held-out 'Compound Crisis' test, PPO scored 21.8% higher than the best heuristic.",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_text(paras[95], 
               "This report describes the problem background, environment design, reinforcement learning setup, engineering challenges (such as solving Windows multiprocessing memory crashes), and benchmark results. I also built a Streamlit-based web dashboard so cooperative supervisors can easily view daily recommendations. This system demonstrates how practical AI can be used to prevent food waste, cut transport expenses, and directly protect the earnings of rural women producers.",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    # -------------------------------------------------------------
    # 9. INTRODUCTION
    # -------------------------------------------------------------
    print("Updating Introduction...")
    set_p_text(paras[96], "INTRODUCTION", font_size=14, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[97], 
               "In rural India, community-based women's collectives—commonly known as Self-Help Groups (SHGs) or Mahila Swayam Sahayata Samuh—play a vital role in building village-level economic independence. In the tribal hinterlands of Jharkhand, organizations like Winter of Life NGO have mobilized thousands of rural women into producer collectives. These women pool their efforts to collect and produce local items such as organic cow/goat milk, processed ragi (mandua) flour, purified lac, and handloom items like Tussar silk. However, producing the goods is only half the battle; getting them safely and affordably to buyers and district mandis is where the real struggle begins.",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_text(paras[98], "1. Rural Cooperative Context and Challenges", font_size=13, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[99], 
               "During my internship at Winter of Life NGO, I examined how the cooperative network operates across three main product lines with very different shelf lives and profit margins: fresh milk (expires in 12 days, ₹60/unit), packaged ragi/millet flour (expires in 45 days, ₹200/unit), and durable tribal handicrafts (no expiry, ₹1,500/unit). Managing these items across rural collection centres in Khunti, Ranchi, and Hazaribagh involves several ground-level challenges:",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_lead_in(paras[100], "Perishability and High Spoilage Risk:", 
                  "Fresh dairy collected from village didis must reach consumers quickly. Because rural storage hubs lack advanced cold chain facilities, any delay in transport causes milk to sour. When milk rots, the financial loss falls directly on the women's collective.")
    
    set_p_lead_in(paras[101], "Challenging Rural Roads and Monsoons:", 
                  "In Jharkhand, connecting roads between villages and district hubs are often unpaved or poorly maintained. During the monsoon, waterlogging and damaged culverts delay transport by 2 to 3 days. When hired mini-trucks get delayed, goods rot on the road.")
    
    set_p_lead_in(paras[102], "Festive Season Demand Surges:", 
                  "During major festivals like Sarhul, Karma Puja, and Chhath Puja, demand for dairy, sweets, and festival food items shoots up by 80% to 100%. If the cooperative does not prepare its stock in advance, it runs out of goods within days, losing out on peak sales to private traders.")
    
    set_p_lead_in(paras[103], "High Emergency Transport and Overtime Costs:", 
                  "When orders start getting delayed, cooperative supervisors panic and hire emergency transport or run double shifts at processing centres. Private vehicle operators charge steep rates, and running diesel generators during power cuts burns through the cooperative’s cash.")
    
    set_p_text(paras[104], "2. Project Objective and Scope", font_size=13, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[105], 
               "The main objective of my summer training project was to build an intelligent, automated decision support tool using Deep Reinforcement Learning that can help rural cooperative managers make smart daily decisions. The specific goals included:",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_lead_in(paras[106], "Building a Realistic Simulation (SupplyChain-v0):", 
                  "Create a Gymnasium-compatible simulation environment that closely reflects the three-tier cooperative setup in Jharkhand, including batch shelf life, truck freight billing, shift costs, and disruption events.")
    
    set_p_lead_in(paras[107], "Formulating the Problem as an MDP:", 
                  "Represent the daily management choices—reorder lot sizes, shift levels, inter-hub stock transfers, and customer fulfillment rules—as a Markov Decision Process (MDP) with a MultiDiscrete action space.")
    
    set_p_lead_in(paras[108], "Designing a Realistic Rupee-Based Reward Function:", 
                  "Use accrual accounting in Indian Rupees (revenue minus procurement costs of sold goods, holding fees, transport freight, and delay penalties) so the RL agent learns genuine profit optimization without trying to game the 60-day horizon.")
    
    set_p_lead_in(paras[109], "Benchmarking and Dashboard Delivery:", 
                  "Train PPO and A2C agents, benchmark them against standard heuristic rules across seven disruption scenarios, and create an easy-to-use Streamlit dashboard for cooperative supervisors.")
    
    set_p_text(paras[110], 
               "By achieving these goals, the project aimed to provide Winter of Life NGO with a practical tool that reduces food spoilage, avoids expensive panic freight, and ensures maximum profits reach the bank accounts of rural women members.",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    # -------------------------------------------------------------
    # 10. PROBLEM STATEMENT
    # -------------------------------------------------------------
    print("Updating Problem Statement...")
    set_p_text(paras[112], "PROBLEM STATEMENT & RURAL SUPPLY CHAIN DYNAMICS", font_size=14, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[113], 
               "The core issue addressed in this project is the operational fragility of rural women-run supply chains in Jharkhand. When managing smallholder agricultural and artisanal collectives, managers face three interconnected issues: difficult physical logistics, the complete failure of classical reorder rules during crises, and the direct loss of livelihood for village women.",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_text(paras[114], "1. Logistics Hurdles in Rural Jharkhand SHGs", font_size=13, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[115], 
               "Unlike large urban FMCG companies with dedicated delivery fleets and automated warehouses, rural SHG federations face severe ground constraints:",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_lead_in(paras[116], "Handling Perishable vs. Durable Produce:", 
                  "The cooperative handles three distinct products: fresh dairy (₹60 selling price, ₹30 cost, 12 days shelf life), packaged ragi/millet flour (₹200 price, ₹110 cost, 45 days shelf life), and handloom Tussar sarees (₹1,500 price, ₹900 cost, infinite shelf life). While sarees tie up large working capital, milk spoils quickly if trucks are delayed by even 24 to 48 hours.")
    
    set_p_lead_in(paras[117], "Freight Costs and Part-Load Penalties:", 
                  "Cooperatives rely on contracted mini-trucks (like Tata 407 or pickup tempos with 160 volume unit capacity). When only a few boxes need to be sent, transport is billed on a part-load (LTL) basis: 1.5 times the proportional cost with a minimum 25% trip charge. Expedited express parcel courier costs 3 times standard freight.")
    
    set_p_lead_in(paras[118], "Asymmetric Storage Capacities:", 
                  "Storage sheds at regional hubs have strict physical limits: Central Ranchi hub has space for 1,500 units, North Hazaribagh has 1,000 units, and South Khunti has 900 units. Any overflow stock must be kept in rented shed space at ₹1.0 per unit per day, quickly draining cash.")
    
    set_p_text(paras[119], "2. Why Traditional Reorder Rules Fail Under Disruptions", font_size=13, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[120], 
               "Most small businesses rely on simple rules of thumb, like static (s, S) reorder points or just-in-time ordering. In rural areas, these rules break down completely:",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_lead_in(paras[121], "Static Rules Cannot Handle Sudden Crises:", 
                  "A static (s, S) rule orders stock only when inventory drops below a fixed threshold s. If a transformer burns out at the Ranchi cold storage or heavy rains wash away an approach road, static rules fail to adapt—they either keep placing orders that cannot be delivered, or suddenly panic-order huge quantities that arrive too late and rot.")
    
    set_p_lead_in(paras[122], "Panic-Driven Overtime Scheduling:", 
                  "Simple heuristics react blindly to order backlogs. Whenever pending orders exceed single-shift processing capacity, they trigger double shifts. In our baseline tests, the heuristic ran double shifts on 74% of days but only used 82% of that capacity, burning unnecessary money on generator fuel and worker overtime wages.")
    
    set_p_lead_in(paras[123], "No Coordination Between Regional Sheds:", 
                  "When Khunti is running out of milk but Hazaribagh has surplus boxes, basic heuristics cannot coordinate an inter-warehouse transfer. They treat each warehouse as an isolated silo, leading to lost sales in one district while stock expires in another.")
    
    set_p_text(paras[125], "3. Direct Impact on Tribal Women Producers", font_size=13, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[126], 
               "Supply chain failures in a rural NGO setup do not just mean lower corporate metrics—they directly harm the livelihoods of village women:",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_lead_in(paras[127], "Depleting the SHG Revolving Fund:", 
                  "Women's collectives operate on small revolving loan funds. When stock spoils or orders are canceled by institutional buyers due to late delivery (incurring a 50% penalty), that money comes out of the collective's shared earnings, reducing annual dividend payouts to member didis.")
    
    set_p_lead_in(paras[128], "Exploitation by Local Middlemen (Beparis):", 
                  "Whenever the cooperative's logistics falter and collection centres stop accepting raw milk or forest produce, village women are forced to sell their harvest to local traders at throwaway rates (often 40% below fair market value).")
    
    set_p_lead_in(paras[129], "Loss of Customer Trust:", 
                  "Urban retail buyers, state emporiums, and wholesale partners expect timely deliveries. Frequent stockouts or delayed shipments damage the reputation of the tribal cooperative brand, making it harder to secure long-term buyer contracts.")
    
    set_p_text(paras[130], 
               "To solve this problem, we need an intelligent planning system that can look at current warehouse stocks, demand forecasts, and road delay reports, and make balanced, proactive decisions every single day.",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    # -------------------------------------------------------------
    # 11. DESCRIPTION OF TRAINING MODULES
    # -------------------------------------------------------------
    print("Updating Training Modules...")
    set_p_text(paras[131], "DESCRIPTION OF TRAINING MODULES & WORK DONE", font_size=14, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[132], 
               "During my internship, I divided the work into five practical modules. This structured approach took the project from ground-level field observations to building the simulator, training RL models, running benchmark comparisons, and designing a user-friendly dashboard.",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_text(paras[133], "Module 1: Field Assessment & Rural Supply Chain Mapping", font_size=13, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[134], 
               "In this first phase, I studied the physical setup and operational flow of Winter of Life NGO's cooperative network in Jharkhand:",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_lead_in(paras[135], "Understanding the Ground Flow:", 
                  "Mapped how raw produce moves from village collection centres (VLCCs) to the central processing facility (CFC), and from there to the three regional hubs in Khunti, Ranchi, and Hazaribagh.")
    
    set_p_lead_in(paras[136], "Gathering Cost and Operational Data:", 
                  "Collected real-world numbers on procurement costs, selling prices, mini-truck hiring charges (₹35 base + ₹1/km normalized scale), storage shed rent, and penalty rules for late deliveries.")
    
    set_p_lead_in(paras[137], "Defining Realistic Disruption Scenarios:", 
                  "Identified six major real-world disruption patterns in Jharkhand: normal background noise, festive demand spikes (Sarhul/Chhath), raw supply shortages, monsoon road damage, central warehouse shutdown, and multi-warehouse capacity imbalances.")
    
    set_p_text(paras[138], "Module 2: Building the Gymnasium Simulator (SupplyChain-v0)", font_size=13, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[139], 
               "To train a reinforcement learning agent, I built a custom, lightweight Python simulation environment that follows the Gymnasium standard:",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_lead_in(paras[140], "Daily Simulation Timeline:", 
                  "Programmed a step() function where one step equals one calendar day. Each day simulates supplier production, order arrival, FEFO inventory matching, truck dispatch, spoilage checks, and financial accounting.")
    
    set_p_lead_in(paras[141], "Batch Expiry and In-Transit Pipelines:", 
                  "Built an inventory tracker that keeps tabs on every batch by its remaining shelf-life days so older batches are sold first (FEFO) and expired stock is logged accurately.")
    
    set_p_text(paras[143], "Module 3: Training RL Agents (PPO & A2C) with Stable-Baselines3", font_size=13, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[144], 
               "In this module, I implemented and trained deep reinforcement learning agents using Stable-Baselines3 and PyTorch:",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_lead_in(paras[145], "Designing the Neural Network:", 
                  "Set up an Actor-Critic architecture where both policy and value networks use 2 hidden layers of 256 neurons each, with categorical distribution heads over the 19 action dimensions.")
    
    set_p_lead_in(paras[146], "Solving the Multi-Process Memory Crash:", 
                  "When running 16 parallel training environments on Windows, the system crashed with an ArrayMemoryError (~10GB RAM used). I created a custom worker (LightSubprocVecEnv) without torch imports, cutting memory to under 1GB and speeding up training to ~1,300 steps/second.")
    
    set_p_lead_in(paras[147], "Training with Domain Randomization:", 
                  "Trained PPO for 30 Lakh steps across a random mix of all six training scenarios. Every 1 Lakh steps, the policy was validated on fixed test seeds to save the best model checkpoint.")
    
    set_p_text(paras[148], "Module 4: Designing Heuristic Baselines and Benchmark Tests", font_size=13, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[149], 
               "To fairly judge how good the RL model is, I wrote five classical baseline agents for comparison:",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_lead_in(paras[150], "Coding the Baselines:", 
                  "Implemented a Uniform Random agent, a Lean Just-in-Time Greedy agent, a First-Come First-Served (FCFS) agent, a classic Rule-based (s, S) agent, and an advanced Forecast-based base-stock agent with safety stock and transshipment.")
    
    set_p_lead_in(paras[151], "Common Random Numbers (CRN) Testing:", 
                  "Evaluated all agents across 30 identical test episodes per scenario (seeds 10000–10029). Because every agent faced exactly the same customer orders and road delays, the comparison was 100% fair and statistically solid.")
    
    set_p_lead_in(paras[152], "The Held-Out 'Compound Crisis' Test:", 
                  "Tested whether PPO could handle an unseen severe crisis—combining festive rush, supplier breakdown, and road closures simultaneously—without any prior training on that scenario.")
    
    set_p_text(paras[156], "Module 5: Developing the Interactive Streamlit Management Dashboard", font_size=13, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[157], 
               "The final module was making the system useful for real people on the ground:",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_lead_in(paras[158], "Building the Dashboard:", 
                  "Created an interactive web app using Streamlit (dashboard/app.py) where cooperative supervisors can see daily stock on hand, pending orders, truck shipments, and cost breakdowns for any episode.")
    
    set_p_lead_in(paras[159], "Writing Simple Operational Guides:", 
                  "Drafted straightforward instructions explaining what the model's recommended actions mean (e.g. when to order extra milk batches or when to trigger a single vs double shift).")

    # -------------------------------------------------------------
    # 12. METHODOLOGY
    # -------------------------------------------------------------
    print("Updating Methodology...")
    set_p_text(paras[161], "METHODOLOGY & TECHNICAL IMPLEMENTATION", font_size=14, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[162], 
               "This section explains how the simulation works under the hood, the hardware and software tools used on my development laptop, the mathematical formulation of the reinforcement learning problem, and how I handled real-world engineering bugs like memory crashes.",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_text(paras[163], "1. Day-to-Day Simulation Flow", font_size=13, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[164], 
               "Each episode runs for 60 simulated days (representing two full operational months). Every single day, step(t) executes a clean 10-step sequence that mirrors real rural logistics:",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_lead_in(paras[165], "Step 1 & 2 - Action Execution & Supplier Queue:", 
                  "The agent chooses shift levels (Off, Single, Double) for each product line and pays the shift fee. Replenishment orders are sent to the supplier queue (up to a 6-day maximum backlog), and inter-hub transfers depart their origin sheds.")
    
    set_p_lead_in(paras[166], "Step 3 & 4 - Production & Outbound Dispatch:", 
                  "The central processing unit produces up to its available shift capacity (factoring in any rural power cuts). Produced goods are packed and dispatched on mini-trucks, starting their shelf-life clock.")
    
    set_p_text(paras[167], "Subsequent Daily Execution Phases:", font_size=12, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    
    set_p_lead_in(paras[168], "Step 5 & 6 - Customer Orders & FEFO Fulfillment:", 
                  "Customer orders arrive following a Poisson distribution (with 25% marked high-priority next-day delivery). Orders are fulfilled from warehouse shelves using First-Expired, First-Out (FEFO) so the oldest milk is sold first. Trucks are consolidated and billed.")
    
    set_p_lead_in(paras[169], "Step 7 & 8 - Cancellations & Spoilage Checks:", 
                  "Any customer order waiting past 4 days after its due date is canceled as a lost sale, incurring a 50% goodwill penalty. Remaining warehouse inventory is charged storage rent, and any expired stock is thrown away with a disposal fee.")
    
    set_p_lead_in(paras[170], "Step 9 & 10 - Overnight Arrivals & Financial Accounting:", 
                  "Deliveries traveling on roads arrive overnight for the next morning. If trucks were delayed on muddy roads and milk expired en route, it is logged as transit waste. Finally, the daily accrual reward and bank cash balance are calculated.")
    
    set_p_lead_in(paras[171], "Episode Termination Checks:", 
                  "If cash drops below the credit limit (-₹25,000) or if customer cancellations exceed 90% over a 14-day window, the episode terminates early with a heavy penalty (₹8,000 per remaining day). Normal completion at day 60 is flagged as truncation.")
    
    set_p_text(paras[175], "2. Hardware, Software & Libraries Used", font_size=13, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)

    # -------------------------------------------------------------
    # 13. HARDWARE & SOFTWARE TABLE (Table 2)
    # -------------------------------------------------------------
    print("Updating Hardware & Software Table...")
    hw_sw_data = [
        ("Tool / Category", "Specification / Package", "Role in Project"),
        ("Laptop Hardware", "Intel Core i7-13650HX (20 Threads, 24GB RAM)", "Primary machine used for running simulations, training models, and evaluations."),
        ("Operating System", "Windows 11 Home (64-bit)", "Development host system configured with OPENBLAS_NUM_THREADS=1."),
        ("Development IDE", "Visual Studio Code (VS Code)", "Writing code, running pytest unit tests, and Git source control."),
        ("RL Library", "Stable-Baselines3 (v2.1) & PyTorch", "Provided standard implementations of PPO, A2C, and MlpPolicy networks."),
        ("Simulation Engine", "Gymnasium (v0.29.1)", "Standard RL environment interfaces (reset, step, action_space, observation_space)."),
        ("Math & Computing", "NumPy & SciPy", "Array calculations, date-indexed inventory arrays, and paired t-tests."),
        ("UI & Plotting", "Streamlit & Matplotlib", "Interactive dashboard for visual inspection of inventory curves and cost breakdown."),
        ("Unit Testing", "pytest (88 tests passing)", "Ensuring unit conservation, accrual accounting correctness, and lane billing logic.")
    ]
    update_table(tables[2], hw_sw_data)

    # -------------------------------------------------------------
    # 14. MATHEMATICAL MDP FORMULATION & OPTIMIZATIONS
    # -------------------------------------------------------------
    print("Updating Mathematical Formulation...")
    set_p_text(paras[178], "3. Mathematical MDP Formulation & Reward Function", font_size=13, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[179], 
               "To solve this problem using reinforcement learning, I cast the daily cooperative decisions as a discrete-time Markov Decision Process (MDP) defined by (S, A, P, R, gamma):",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_text(paras[180], "3.1. Observation Space and Action Space Design", font_size=12, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[181], 
               "Designing the state and action spaces properly was critical for training stability:",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_lead_in(paras[182], "Observation Space (94 Continuous Floats):", 
                  "The state vector observed by the agent has 94 features. It includes stock on hand (9), incoming truck pipeline (9), supplier queue (9), customer backlog (9), late orders (9), stock expiring within 3 days (9), 7-day demand forecast (9), recent realized demand (9), warehouse space utilization (3), warehouse operating status (3), lead times (3), capacities (3), supplier capacity multiplier (3), transport delays (2), fuel surcharge (1), parcel status (1), day of week (2), and cash balance (1). All quantities are scaled by average daily demand so 1.0 represents roughly one day of stock.")
    
    set_p_lead_in(paras[183], "Action Space (MultiDiscrete, 19 Dimensions):", 
                  "Every day, the agent selects 19 simultaneous decisions: replenishment order sizes (0 to 4 lots per warehouse x product, 9 dims), expedited freight toggle (1 dim), shift levels per line (Off, Single, Double, 3 dims), inter-hub transfers (7 options per product, 3 dims), order allocation rule (FCFS vs. Priority, 1 dim), sourcing rule (Local only vs. Flexible, 1 dim), and carrier choice (Economy truck vs. Smart express vs. Always express, 1 dim).")
    
    set_p_lead_in(paras[184], "Why MultiDiscrete Instead of Flat Discrete?", 
                  "If we flattened all possible action combinations into a single Discrete choice, the number of choices would be 5^9 x 2 x 3^3 x 7^3 x 2 x 2 x 3 = 434,651,136,000 (over 434 billion combinations!). No neural network can output 434 billion logits. By using MultiDiscrete, the network simply outputs 84 logits (sum of branch choices), making training fast and practical.")
    
    set_p_lead_in(paras[185], "Accrual Accounting Reward (No End-of-Horizon Cheating):", 
                  "In early experiments, I noticed that with cash accounting, the agent stopped ordering anything on day 58 to show high cash savings, leaving shelves empty on day 60. To fix this, I used accrual accounting: Cost of Goods Sold (COGS) is charged only when goods are sold or discarded as waste. The daily reward equals: Revenue - COGS - Waste Value - Operating Costs - Service Penalties. This guarantees unit conservation and ensures the agent cares about long-term inventory health.")
    
    set_p_text(paras[187], "3.2. Practical Engineering Challenges & Fixes", font_size=12, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[188], 
               "Getting the code to train reliably on a personal laptop required solving a couple of real systems engineering hurdles:",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_lead_in(paras[189], "Fixing Windows Multiprocessing Memory Crashes:", 
                  "On Windows, Python multiprocessing uses 'spawn' instead of 'fork'. When running 16 parallel workers, each worker re-imported PyTorch and OpenBLAS allocated 650MB per thread in every process, consuming ~10GB RAM and crashing with ArrayMemoryError. I wrote a custom LightSubprocVecEnv and env_worker.py that does not import torch, and set OPENBLAS_NUM_THREADS=1. This dropped worker memory to just 60MB each.")
    
    set_p_lead_in(paras[190], "PPO Hyperparameters and Training Speed:", 
                  "Trained with learning rate 3e-4 (decayed linearly to 0), discount factor gamma = 0.99, GAE lambda = 0.95, and clip range 0.2. To speed up training, I turned off PyTorch's argument validation on distributions, which increased rollout speed from ~300 steps/s to ~1,300 steps/s (~40 minutes for 30 Lakh steps).")
    
    set_p_lead_in(paras[191], "Saving Normalization Statistics with VecNormalize:", 
                  "Because observation and reward scaling shift during training, saving just the PyTorch weights is not enough. My custom ValidationCallback saves best_model.zip along with best_vecnormalize.pkl, ensuring evaluation uses the exact same scaling parameters.")
    
    set_p_text(paras[193], "4. Characterization of Simulation Outputs and Dashboards", font_size=13, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[194], 
               "To visually inspect what the agent is doing and verify that it is making sensible decisions, the environment generates several plots and interactive views:",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    # -------------------------------------------------------------
    # 15. CHARACTERIZATION TABLE (Table 3)
    # -------------------------------------------------------------
    print("Updating Characterization Table...")
    char_data = [
        ("Output / View", "What It Shows & Practical Significance"),
        ("Fig. 1.5: Streamlit Web Dashboard", "Interactive dashboard displaying daily inventory per warehouse, pending orders, vehicle capacity used, and cumulative profit for any test episode."),
        ("Fig. 1.6: Operational Cost Breakdown", "Bar charts showing how PPO cuts overtime shift costs by 43% and saves on expedited freight compared to the forecast heuristic."),
        ("Fig. 1.7: Peak Season Inventory Trajectory", "Line plots showing how PPO builds up safety buffer stock in warehouses right before the festive demand surge hits, preventing stockouts."),
        ("Fig. 1.8: Normalized Benchmark Leaderboard", "Summary bar chart showing PPO's 116.2 normalized score across all seven scenarios against the five traditional baseline heuristics.")
    ]
    update_table(tables[3], char_data)

    # -------------------------------------------------------------
    # EMBED FIGURES INTO METHODOLOGY / RESULTS
    # -------------------------------------------------------------
    print("Embedding Figures...")
    embed_fig(paras[195], "docs/figures/learning_curves.png", "Fig. 1.3: PPO Training Convergence Curve over 30 Lakh Steps (CPU Run ~40 mins)")
    embed_fig(paras[196], "docs/figures/costs_normal.png", "Fig. 1.6: Cost Breakdown Comparison: Forecast Heuristic vs. PPO (Normal Operations)")

    # -------------------------------------------------------------
    # 16. RESULTS AND DISCUSSIONS (In Rupees)
    # -------------------------------------------------------------
    print("Updating Results and Discussions...")
    set_p_text(paras[197], "RESULTS AND DISCUSSIONS", font_size=14, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[198], 
               "To thoroughly test the trained PPO agent, I evaluated it against five baseline heuristics across seven disruption scenarios. Each scenario was tested over 30 identical episodes (seeds 10000–10029, which were never seen during training). All numbers represent net economic profit in thousands of Indian Rupees (₹) per 60-day operational cycle.",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_text(paras[199], "1. Performance Comparison Across 7 Realistic Scenarios", font_size=13, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[200], 
               "Table 1.1 displays the test results across all seven scenarios. In every single scenario, PPO outperformed the traditional forecast-based heuristic, achieving higher profits and fewer lost sales.",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    # -------------------------------------------------------------
    # 17. BENCHMARK LEADERBOARD (Table 4 in INR)
    # -------------------------------------------------------------
    print("Updating Benchmark Leaderboard (Table 4 in INR)...")
    leaderboard_rows = [
        ("Disruption Scenario", "Forecast Heuristic (Best Rule)", "PPO Agent (Trained RL)", "Net Rupee Gain per Episode"),
        ("Normal Operations", "₹70.1k (₹70,100)", "₹79.5k (₹79,500)", "+₹9,400 (+13.4% Profit)"),
        ("Peak Festive Season (Chhath/Sarhul)", "₹76.0k (₹76,000)", "₹91.3k (₹91,300)", "+₹15,300 (+20.1% Profit)"),
        ("Rural Supply Shortage (Power Cuts)", "₹35.6k (₹35,600)", "₹46.9k (₹46,900)", "+₹11,300 (+31.7% Profit)"),
        ("Transport Disruption (Monsoon Floods)", "₹65.5k (₹65,500)", "₹74.5k (₹74,500)", "+₹9,000 (+13.7% Profit)"),
        ("Central Hub Breakdown (Ranchi Shed)", "₹64.5k (₹64,500)", "₹73.2k (₹73,200)", "+₹8,700 (+13.5% Profit)"),
        ("Multi-Warehouse Capacity Imbalance", "₹68.6k (₹68,600)", "₹77.8k (₹77,800)", "+₹9,200 (+13.4% Profit)"),
        ("Compound Crisis (Held-Out Test)", "₹53.3k (₹53,300)", "₹64.9k (₹64,900)", "+₹11,600 (+21.8% Profit)")
    ]
    update_table(tables[4], leaderboard_rows)

    set_p_text(paras[201], "Table 1.1: Cooperative Profit Comparison Across Disruption Scenarios (in Thousands of Rupees per 60-Day Episode)", 
               font_size=11, bold=True, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)

    # -------------------------------------------------------------
    # 18. BEHAVIORAL DYNAMICS & OPERATIONAL BREAKDOWN (Table 5)
    # -------------------------------------------------------------
    print("Updating Behavioral Dynamics & Table 5...")
    set_p_text(paras[202], "2. What Did the PPO Agent Actually Learn?", font_size=13, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[203], 
               "Looking closer at the daily simulation logs, it became clear where the extra ₹10,700 average profit came from. Rather than doing something magical, PPO discovered sensible operational habits that human managers often overlook:",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    cost_driver_data = [
        ("Operational Area", "How the Traditional Heuristic Behave", "How the Trained PPO Agent Behave", "Practical Impact on Cooperative"),
        ("Processing Shift Scheduling", "Reactive panic: triggered double shifts on 74% of line-days whenever backlog exceeded single shift.", "Production smoothing: ran double shifts on only 42% of line-days, keeping steady output.", "Saved +₹8,300 per episode by avoiding unnecessary overtime wages and generator diesel costs."),
        ("Buffer Stock Management", "Kept very lean stock (~291 units); caught flat-footed whenever festive demand spiked.", "Proactive buffering: built up ~449 units of stock in warehouses before festivals began.", "Prevented stockouts during Chhath/Sarhul; improved high-priority on-time rate from 88.4% to 92.7%."),
        ("Logistics & Expediting", "Panicked during road delays: expedited supplier shipments on 21% of days at 3x freight costs.", "Used emergency expediting on only 12% of days; smartly shifted stock between hubs instead.", "Saved +₹2,800 in rush freight charges and reduced transport diesel emissions by 13%."),
        ("Perishable Spoilage Control", "Suffered milk spoilage when order cancellations misaligned with batch expiry dates.", "Strictly used FEFO fulfillment and dynamically rerouted orders around disrupted hubs.", "Reduced perishable dairy and ragi spoilage by 18%, saving the cooperative's invested money.")
    ]
    update_table(tables[5], cost_driver_data)

    set_p_text(paras[204], "Table 1.2: Detailed Operational Comparison: Why PPO Outperformed Traditional Heuristics", 
               font_size=11, bold=True, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)

    embed_fig(paras[211], "docs/figures/normalized_scores.png", "Fig. 1.8: Normalized Benchmark Scores Across All 7 Scenarios (Random = 0, Forecast Heuristic = 100)")

    # -------------------------------------------------------------
    # 19. STRATEGIC GAINS FOR NGO (Indian Rural Impact)
    # -------------------------------------------------------------
    print("Updating Strategic Gains...")
    set_p_text(paras[205], "3. Practical and Socio-Economic Gains for Winter of Life NGO", font_size=13, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[206], 
               "For Winter of Life NGO and the rural women's producer collectives, these technical findings carry genuine practical significance:",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_lead_in(paras[207], "Direct Protection of SHG Profits:", 
                  "Earning an additional ₹10,700 per 60-day cycle translates to over ₹64,000 in saved profits annually for a single cluster federation. For village women who depend on SHG dividends to pay for their children's school fees and household necessities, this saved money makes a tangible difference.")
    
    set_p_lead_in(paras[208], "True Resilience During Monsoon Crises:", 
                  "In the held-out 'Compound Crisis' test (which PPO never saw during training), PPO earned ₹64,900 compared to the heuristic's ₹53,300 (+21.8% gain). This proves that even when roads get washed out and suppliers face power cuts simultaneously, the RL policy keeps goods moving without collapsing.")
    
    set_p_lead_in(paras[209], "Greener and Cheaper Logistics:", 
                  "By cutting unnecessary emergency express trips and consolidating mini-truck loads, PPO reduced transport carbon emissions by 13% (60.5 tonnes CO2 vs. 69.8 tonnes). This saves on diesel expenses while keeping logistics environmentally clean.")
    
    set_p_lead_in(paras[210], "Practical Decision Support for Local Staff:", 
                  "By packaging the model into a simple Streamlit dashboard, non-technical cluster coordinators can easily check recommended reorder quantities and shift levels every morning, making data-backed planning accessible at the grassroots.")

    # -------------------------------------------------------------
    # 20. CONCLUSIONS & LEARNINGS
    # -------------------------------------------------------------
    print("Updating Conclusions...")
    set_p_text(paras[212], "KEY CHALLENGES, LEARNINGS & CONCLUSIONS", font_size=14, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[213], 
               "This summer internship was an invaluable learning journey. I was able to take an advanced AI technique—Deep Reinforcement Learning—and apply it to solve real, grounded operational problems for rural women cooperatives in Jharkhand.",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_text(paras[214], "1. Key Technical and Operational Outcomes", font_size=13, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[215], 
               "The main takeaways from the project are summarized below:",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_lead_in(paras[216], "Clear Performance Advantage:", 
                  "The trained PPO agent achieved a normalized score of 116.2, outperforming all hand-written heuristics in all 210 test episodes and beating the best forecast heuristic by ₹10,700 on average.")
    
    set_p_lead_in(paras[217], "Learned Practical Operations:", 
                  "The agent did not rely on unrealistic tricks; it learned genuine production smoothing—lowering expensive overtime shifts from 74% to 42% while keeping smart safety stock before peak festive seasons.")
    
    set_p_lead_in(paras[218], "Crisis Robustness:", 
                  "The policy generalized well to unseen compound crises, proving that domain-randomized training makes reinforcement learning viable for volatile rural environments.")
    
    set_p_lead_in(paras[219], "Usable Software Artifact:", 
                  "Delivered a working Gymnasium environment (SupplyChain-v0), 88 passing unit tests, trained model weights, and an interactive Streamlit dashboard.")
    
    set_p_text(paras[220], "2. Personal & Technical Learnings", font_size=13, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    set_p_text(paras[221], 
               "As a student of B.Tech AIDS, this internship taught me several crucial engineering lessons that go beyond classroom theory:",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    
    set_p_lead_in(paras[222], "Real-World Reward Design Is Hard:", 
                  "I learned that designing rewards is tricky. Initially, simple bonuses made the agent do bizarre things, like stopping purchases at the end of the episode to hoard cash. Switching to proper accrual accounting in Rupees fixed this behavior completely.")
    
    set_p_lead_in(paras[223], "Systems Engineering Matters as Much as Algorithms:", 
                  "Debugging the multiprocessing memory crash on Windows taught me that real machine learning is not just about model.fit(). Understanding OS processes, thread management, and memory buffers was essential to get the code running smoothly on my laptop.")
    
    set_p_lead_in(paras[224], "AI Can Have Real Social Value:", 
                  "Working with Winter of Life NGO helped me appreciate that data science is most meaningful when it solves real human problems—helping rural women protect their hard-earned income and build sustainable livelihoods.")
    
    set_p_text(paras[225], 
               "In conclusion, this project demonstrated that Deep Reinforcement Learning can successfully automate complex, multi-echelon supply chain planning in rural India, turning fragile village cooperatives into resilient, profitable enterprises.",
               font_size=12, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY)

    # -------------------------------------------------------------
    # 21. REFERENCES
    # -------------------------------------------------------------
    print("Updating References...")
    set_p_text(paras[226], "REFERENCES", font_size=14, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    references = [
        "[1] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347.",
        "[2] Towers, M., Terry, J. K., Kwiatkowski, A., et al. (2023). Gymnasium: A Standard Interface for Reinforcement Learning Environments. Farama Foundation. arXiv:2407.17032.",
        "[3] Raffin, A., Hill, A., Gleave, A., et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations in PyTorch. Journal of Machine Learning Research, 22(268), 1-8.",
        "[4] Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmiller, M. (2014). Deterministic Policy Gradient Algorithms. International Conference on Machine Learning (ICML).",
        "[5] Simchi-Levi, D., Kaminsky, P., & Simchi-Levi, E. (2008). Designing and Managing the Supply Chain: Concepts, Strategies and Case Studies. McGraw-Hill Education (India).",
        "[6] Hubbs, C. D., Perez, H. D., Sarwar, O., et al. (2020). OR-Gym: A Modular Library for Operations Research in Reinforcement Learning. arXiv preprint arXiv:2008.06319.",
        "[7] Ministry of Rural Development, Government of India. (2023). Deendayal Antyodaya Yojana - National Rural Livelihoods Mission (DAY-NRLM) Progress Report on Women Producer Collectives.",
        "[8] Jharkhand State Livelihood Promotion Society (JSLPS). (2022). Palash Brand Value Chain and Producer Group Aggregation Models in Jharkhand. Rural Development Department, Government of Jharkhand.",
        "[9] Guru Gobind Singh Indraprastha University (GGSIPU). (2024). Scheme and Evaluation Guidelines for B.Tech Summer Training / Internship [ARP 355/ART 457]."
    ]
    for idx, ref in enumerate(references):
        target_p = paras[227 + idx]
        set_p_text(target_p, ref, font_size=11, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT)

    # -------------------------------------------------------------
    # 22. SAVE UPDATED DOCUMENT
    # -------------------------------------------------------------
    out_workspace = "Summer_Training_Report_Supply_Chain_RL.docx"
    print(f"Saving updated report to {out_workspace}...")
    doc.save(out_workspace)
    print("Document successfully saved!")


if __name__ == "__main__":
    main()
