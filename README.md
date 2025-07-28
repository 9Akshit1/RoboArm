# RoboArm
DESCRIPTION: RoboArm is just like how it sounds. It is a robotic arm with 3 joints that can mimic real arm movements. It uses small servo motors controlled by an EMG sensor to move based on brain signals. The hand can also be controlled manually via a computer or preset motions for easy testing and flexibility.

INSPIRATION: I've always been inspired by how human arms move and wanted to recreate that motion using motors and tendons. Additionally, I rpeviosuly buitl a RoboHand so to add on to that project, I wanted to add a RoboArm to extend it. 

Link to RoboHand: https://github.com/9Akshit1/RoboHand (they are separate projects)

# Final CAD
The separate part files and body files are in ther CAD folder. The full built hand CAD File is called FULL-RoboArm.stl in the CAD folder. 

Also, please note that I can always just add supports before I 3D print, so we do not need to worry about that!

Additionally, I made two versions of my arm which are the standard robotic arm verison (the type of design you would see when you search up a robotic arm), and the cooler human-like arm version (which was designed after my arm). The human arm version is a little more messy and likely contains a few errors that I need to fix, which is why I hope the grader (or whoever is reading this) can ignore the mistakes in the human arm version. I will first print out the robotic arm version anyways, and only AFTER I fix the human arm version and test it so that it works in the CAD software, will I print it.

The standard robotic arm version:

![Final CAD Robotic Arm Section 1 Separated]()

The cool human arm version:

![Final CAD Human Arm Section 1 Separated]()

This is what the full build hand will look like. I will likely hot glue every piece together. 

The standard robotic arm version:

![Final CAD Robotic Arm]()

The cool human arm version:

![Final CAD Human Arm]()

# Final Circuit Schematic
Schematic is called RoboArm.kicad_sch in the schematic_pcb/RoboArm folder.

![Final Schematic](https://hc-cdn.hel1.your-objectstorage.com/s/v3/09242684435c8adfca3481af5a91649d5c3d4354_schematic_jy25.png)

# I am not using a PCB, because I will manually solder everything.

# Final Firmware Stuff
The firmware software is in the firmware folder, and are called main.py.

![Code Picture]() 

## Bill of Materials (BOM)

| Component | Model/Part Number | Qty | Price (CAD) | Price (USD) | Link | Notes |
|-----------|-------------------|-----|-------------|-------------|------|-------|
| ESP32 DevKit V1 | ESP32-WROOM-32 | 1 | $11.39 | $8.43 | [Link](https://www.amazon.ca/ESP-WROOM-32-Aideepen-30PIN-Development-MicroPython/dp/B0DDPJQX3X/ref=sr_1_19?crid=2UXH2KGFOR3CC&dib=eyJ2IjoiMSJ9.ibEv7cvxgsX2dmbfesfY9usWxppYWccvJc_3h8zgi53kA-4DhFx3Omk510-lTkd8eQOxBVLuxeIw-xoaMcD8LpsI6xuzUp3k0SwjWLKlALsHHcHbi7o5MXZ1jpdsCHQpyEEvX5YOTakwqqJyzCLZJqQdiVOJaYnTWJd3guMW2vgkNfFBwDRO2E6teyZb65qShQo9QZ8R70d-jIia0JVuVLtCuBAS8pezztQvlij5nbhRLnG-h8Y77LbOb-oVlx1qtC2B5W-Ujav0hmBYfoTNifTZpKRpViqK58ri9qUTeYc.Ua8rhpMGX2Z5QwJkuXOw8_YeDogXI-72GWUhaShYS7Q&dib_tag=se&keywords=DEVIKIT+V1+ESP32&qid=1753664090&refinements=p_36%3A-2000&rnid=12035759011&s=electronics&sprefix=devikit+v1+esp32%2Celectronics%2C71&sr=1-19) | Main controller |
| ESP32-CAM | ESP32-CAM | 1 | $17.99 | $13.31 | [Link](https://www.amazon.ca/Aideepen-ESP32-CAM-ESP32-CAM-MB-CH-340G-NodeMCU/dp/B0CMTVFCYD/ref=sr_1_1?crid=3KX92GYL9TYB&dib=eyJ2IjoiMSJ9.xk_W05RHOWbB-2VIcpqhAeAQMviHQXbT5Pgp5PJDeySLfNRc4zthbE66sgqRm00KQbw6Lfaoj8dhOlMG5DqFvZZi5fbKJh2yCqh0uexdYqXf-2vgxAAkmwqIy3NP_QogQ-3QjBs-W1PN747J18wojYnnW9Irq2RTFhByVa77WvmyaE9YmegEa0zxb2XlGtI_YwjrWIOE5SyAIcx2yvOb4cNOhXT5uaF7GOq6IPUmhLSWITDjhOw7wjdvMRsTPhKxb1Us6BDcoBI7o3yIgu1GDT5vGoe3w5lA5oLSETGg7bg.2rrL-0tPXzA070WtanrSRZ8w3XYs1b2VoJOHFJCRK8c&dib_tag=se&keywords=ESP32-CAM&qid=1753664337&refinements=p_36%3A-2000&rnid=12035759011&s=electronics&sprefix=esp32-cam%2Celectronics%2C84&sr=1-1) | Computer vision. The amazon link was the best and cheapest deal I could find |
| MG90S Micro Servos | TowerPro MG90S | 2 | $14.11 | $10.44 | [Link](https://www.amazon.ca/MG90S-Servo-Micro-Geared-Helicopter/dp/B091CMGCMH/ref=sr_1_2?crid=3R8T240FPM1VQ&dib=eyJ2IjoiMSJ9.QcxBBCoz1mvrIJ2muL9-HwYbjuy3ExGHDTPDAPHF8-y9rckP_9GwMAORvC2xDYa8Zv_S3TKyGO45GICHSG9i-cHgqKaZgt3fwSG18ZwqSrMkOSNcrXoN7jfmiDwrTJwp8Z59n8NcN7EoD9VWynzH8GP3CCmWqFgDeYlZtLizlB0zddDLXfAKJEItMSTM7RLrRwEXS4JaX_tyaqYUfFmhvkGj9_Q1g6HL2YVFnvzaD_7yKThU5_7zXUJUsvNy5nIfHQxUrNS9PtggMcGisETyo5rub4sIzvbIsSQkV87JBL0.yOVZhC9LOPBaXD-p7G9CeGlJgdzAXfFeIkLXcY-mmQo&dib_tag=se&keywords=MG90S+servo+motors+2pcs&qid=1753668424&sprefix=mg90s+servo+motors+2pcs%2Caps%2C91&sr=8-2) | Pack of 5, need only 2, but it was the cheapest price. Also, I can use the other extra ones if some of them fail/break |
| Dynamixel Servos | XM430-W350-T | 4 | $89.99 | $66.59 | [ROBOTIS Store](https://www.robotis.us/dynamixel-xm430-w350-r/) | High-torque servos |
| PWM Driver | Adafruit PCA9685 | 1 | $9.99 | $7.39 | [Link](https://www.amazon.ca/Newhail-PCA9685-Channel-Arduino-Raspberry/dp/B08YD8PDLS/ref=sr_1_8?dib=eyJ2IjoiMSJ9.EEfjDa5DorESYqQSBRyN4y2RIrvgp7hjEJIhLqpWWRobb2NvP8jwUd4v9g2FhjxoIoxgB_oDgjq3yQXu7OxuEfj3He0Rn_VfibmrUTx4c_r1VwrveTCrfCPC8S0Zfw9czv33jXJzzMRUZries6c14qp3lMRviSuAptk_b1jlXod62SaH1cCIpxl6rZnfXIAoo8wtoS2tHwRTehMCFRWCTAcCwfCrQKBowUFLwMsxnCA-T5_knIgiWIQLTVXGGpei1k9IdYemFr66Xay6u8_ZT3V-hQqq00l6ALru5NmCBwg.hwhubtqv-EM_IioGbQA7V8yTv8jauMlt7NkJjLB3RH4&dib_tag=se&keywords=pca9685+16+channel&qid=1753665414&refinements=p_36%3A-2000&rnid=12035759011&sr=8-8) | 16-channel PWM |
| Level Shifter | TXS0108E | 1 | $5.89 | $4.36 | [Link](https://www.amazon.ca/10pieces-Compactly-Translators-Supporting-Efficient/dp/B0F6JZL6FR/ref=sr_1_23?crid=YR4I1QV9XSI9&dib=eyJ2IjoiMSJ9.nrGJX6xSrKidUZqvWntdNSdo38IY3cM3LgKAxsAz6fEHv7YcsCyoRdBeD-g6hBTHa4FYcZ6Qxhd-jcvbRXpd_eDhfbkKZDzJuX3N9qmp_2dTxOrswWw4-iwq5fTYT6See2z1GLUrIo0DQoMW3yukU2OHxdWlARBDtWJ303BDgVCIsKQmHBpinOy-es0iHzXxdzyyO0IO6dv6pPkTRUfyCHHmBf27kPkW9LJ6hpqk7gTOEzdBLrZMgQ8qj04FV6yGJeko5jXvdJhTsAQVtVeqNYQRMRnmdG1BM_vxmUN26zA.Y9Iee34-5jZcDP7BPKLKD7cnhiin-FpIVMz7gR-Qwco&dib_tag=se&keywords=txs0108e+1pc&qid=1753665524&sprefix=txs0108e+1pc%2Caps%2C92&sr=8-23) | 3.3V to 5V converter. The amazon link says 10 pcs (though it might be a typo because of how crazy good this deal is. So it could be 1 actually) but they're only for ~$5 which is absolutely the best and most cheapest deal there |
| EMG Sensors | MyoWare | 10 | $47.99 | $35.52 | [SparkFun](https://www.sparkfun.com/products/13723) | Muscle activity sensors |
| ADC Modules | ADS1115 | 3 | $10.39 | $7.69 | [Link](https://www.amazon.ca/ADS1115-Converter-Digital-Development-Precision/dp/B0FGX3FJ62/ref=sr_1_13?dib=eyJ2IjoiMSJ9.P1kddnTm_aeAgBj_1pKscu3GPNfgHM5cf-Za6bs5YpCxW4I6JmSiG3ROWDjQPcKbBX_SvjzT4ZXZ0TsUhhLSU7lg8YMuIzrPTYJtdRbfzEDoyLI92eNmB02YVk_zd4XW0CgfqBPa74teIKb7YgKpVrjNKwLnRM4gcrqKUgP98aE4iTfAmq5CNzIQy8rJMZnyWmFhTdvTfRu1o7kDPhRfLyxJwlRk-tuB-Rpz0zIMlBFiB3nYfmhknV7K1XRH73uKn8GaLzOHPr302XF2rRSAaAMFQjyOzrq5aev952BIkU4.Nxqej9c2pGUCl710kMr1hb8NnowMhOhf92nvUkUw0kU&dib_tag=se&keywords=ads1115&qid=1753666001&sr=8-13) | 16-bit ADC |
| IMU Sensor | BNO055 | 1 | $15.94 | $11.80 | [Link](https://www.amazon.ca/BNO055-Sensors-Acceleration-Gyroscope-Position/dp/B0F37YCHGK/ref=sr_1_2?dib=eyJ2IjoiMSJ9.WLaYBoVdqWjMGSSPr88oO2Nrehx6QDtC_PoOZINWh9G_vT69XLHGcDhFfx37k1gpJkwcLLT192XOFWwsdGhSfGdRvfsHkY5KBFpYFz31yVJG57rG6qQOHFxEq_ZnH4HUBVyZj0r-5gMcZiSRum1drwrtr1U34gjOI9PJ3wUEOgBET7TN8uHzs8zM5JNtxGrTi1NmvTqz7dovXGsNPxZktWil8qMlyQySMjTU5AOXmfcxZCWiUczC242pckUERQdiIUfmNxK-A_iHS9Qcf3AiH7HTBusnrN7knHWiHKQdsI4.LrSlqy9e2DhSZuZ38Ds_FL_6i5qW3vwA7zkiM0AL3tw&dib_tag=se&keywords=bno055&qid=1753666236&sr=8-2) | 9-DOF orientation |
| Power Supply | 5V 20A SMPS | 1 | $20.53 | $15.19 | [Link](https://www.amazon.ca/Adapter-Switching-Universal-Regulated-Security/dp/B09PV8S7ZW/ref=sr_1_28?crid=2EB6WMESHSP37&dib=eyJ2IjoiMSJ9.D0Qgb28rwnkSZBXWB1QKBEM_kYhCqy0SMIF7DQ82rIPFFQAMz640gEt2MXS1xRihEqCNYEri3Jb1oqIJgOoalFszNjYEqFmi7gKpq8CoxGtswz_I96C24cumq4Ova_pVM8uCsGTw6Oa7xr7Cr0SWf4hu3q-LRPYKR21HrvzKFS6Ncsp1qhpXG_wzVb3RewJAp5YOaFDqNDN9K5C8JM7tVBJgyEMWSfgVlacemDQEYJtepnyaU4W4zjNxfuw3fPBXt92StxL6-XhRLliBJXdbbdpIBAkEmcwXT82rhq_lvuk.n8QBOC92yvgjlbUDO6SZuOmU-GYL4k_ad0GULDsdTsA&dib_tag=se&keywords=5v+15a+power+supply+SMPS&qid=1753666400&refinements=p_36%3A-3000&rnid=12035759011&sprefix=5v+15a+power+supply+smps%2Caps%2C92&sr=8-28) | High current supply, is SMPS, and is best & cheapest deal |
| Pull-up Resistors | 4.7kΩ 1/4W | 2 | $3.51 | $2.60 | [Link](https://www.amazon.ca/10PCS-Metal-Film-Resistor-4-7K/dp/B0C7L6QFHN/ref=sr_1_2?crid=3SBZTK2V2RUVU&dib=eyJ2IjoiMSJ9.-rIVY3wSj84NnF-5g4ZEatT6hVOw4KQGy5mgm4MFN3UGRE8-xI7SimasOJRfEIKL-bUV9YrpEgSqXjgH19Xye1ZftrCxiyGV3P0wjpTCIIiiSfDGFdLBg99Y1hfafXmv9Oe7tMuaUxPU9VMOWU2PO5zLpJTy454LYnI7wM7fI7zp01PumPY5t6xVHxe-YKtETJd3LH-yZX5iLm01EIZ9CuygfdbJxticUbk6P3yFAWBV_PqhxyYmrR9vVYDTB1kU-c9myssErRoaz1lpqOkgKFSec_5lvjHxpb_rSfs7r0c.4fv0ql0qhD3EmZr_YuA16C3JQXJYh9Hdq1Q78qYbVj0&dib_tag=se&keywords=4.7k+pull-up+resistor&qid=1753666824&sprefix=4+7k+pulll-up+resistor%2Caps%2C81&sr=8-2) | I²C pull-ups. Cheapest deal. The link has 10 pieces, but there was no product that sold them individually. |
| Capacitor 1000µF | Electrolytic 25V | 1 | $5.88 | $4.35 | [Link](https://www.amazon.ca/Rosrrtlm-1000UF-Electrolytic-Capacitor-10x20mm/dp/B0DDKYDKMJ/ref=sr_1_31?dib=eyJ2IjoiMSJ9.VIXkAEzWdz_HIXo2Ih6CFSw5kZrG_P8wUpko1A8YANSuhJF08DSEGvxqjTNQgbW_DeWZJ2VL6gIa3XWpzTM2jUBVwbQsYTNUkccYiCTEXy7CZqc6QDN-4ZYdb7F35ssTEXL1JbueAGGwTsP-2-oOfS7dDTohu4Xns4G7i7gB2t2wpoyBZp9NKeVbPCQjiDDqO2MQ9LOoThHDKzfAMI1uYxazZE-Twy94dp5PQpHQdgmZ26iwWytiYiUOZGaQIM4ODwovuqcZlu-6sK_bLC70H67leqKgoqTdFJncFbwgDtY.Ltrh2t66WQ0AXSdFZZjR_qD9pnWPpihbZY_D_qHqZhQ&dib_tag=se&keywords=1000uf+capacitor&qid=1753666936&sr=8-31) | Power filtering. The link has the best deal considering shipping costs, because all of the other cheaper ones has ~$10 in shipping compared to this which has Free shipping and is only $2 more expensive. Also, the link has 5 pieces, so I can use the extra since these cheap ones will likely break anyways. |
| Capacitor 470µF | Electrolytic 25V | 1 | $5.54 | $4.10 | [Link](https://www.amazon.ca/Molieeigin-Celsius-Electrolytic-Capacitor-10x20mm/dp/B0BWDZYFHM/ref=sr_1_39?dib=eyJ2IjoiMSJ9.VIXkAEzWdz_HIXo2Ih6CFSw5kZrG_P8wUpko1A8YANSuhJF08DSEGvxqjTNQgbW_DeWZJ2VL6gIa3XWpzTM2jUBVwbQsYTNUkccYiCTEXy7CZqc6QDN-4ZYdb7F35ssTEXL1JbueAGGwTsP-2-oOffKhlsxEYJlBiHwe1SDycYwN3d6alHrH5bj30VJgZIJKeB7UpBkLnlTUm3r3eumMGOJbP45JxGf-PDAgMMMiVOvpnnFTRI8bPmPNjq4EDLXQqwRzgYK2ugYK5WFqxhpgCO7dDwnVAZudkQKw-UvwaHY.nkSuWN35QvZoi8atxMDwhR4CGFflL5Th9lFkJ_hqgsc&dib_tag=se&keywords=470uf+capacitor&qid=1753668023&sr=8-39) | Power filtering. The link has the best deal considering shipping costs, because all of the other cheaper ones has ~$10 in shipping compared to this which has Free shipping and is only $2 more expensive. Also, the link has 5 pieces, so I can use the extra since these cheap ones will likely break anyways. |
| Capacitor 220µF | Electrolytic 25V | 1 | $2.46 | $1.82 | [Link](https://www.amazon.ca/Capacitor-10x-Radial-Electrolytic-Capacitor-5x11mm/dp/B09W1DZCWR/ref=sr_1_2?dib=eyJ2IjoiMSJ9.W54DUFjX7XBwM02IL26afMZjTG2nmQ1qkyy9wnJ52QsPQLDaBmuUm_La_Z0aKxdDOpnPFblSxpqjXjtWfRrQauCGZLUQOua6N-yRa6PkQ7AaCaCReRbEA5ZR8AFSvDZRAaJLdqejbPzAlQMkp7PO0qwik9ujnkqHcpY_9RK53CsB1fG8VRigTbcJu_TVHogUgK5vkBO700NgCW2s1UZZIAWRoR3Hr1p21TnhZlZKMCU1bnyRPTqDC5I8FVVJF23PHRHdpmFVmiJZoh6Xj64nZWUqvbF_L0D0gkkh-fJoQwQ.hnmcwLXv7RkNQ9QL-ixLbZqzxMqP8xd0e8LPet74fM8&dib_tag=se&keywords=220uf+capacitor&qid=1753668164&sr=8-2) | Power filtering. The link is literally the cheapest one there already, with and without the shipping costs. Also, the link has 10 pieces, so I can use the extra since these cheap ones will likely break anyways. |
| Jumper Wires | Dupont Wires | A lot | N/A | N/A | N/A | I already have a lot of these |
| Breadboard | Half-size | 1 | N/A | N/A | N/A | I already have 2 empty ones to use |

## Cost Summary
### Component Totals
- **Subtotal**: $193.59 USD / $261.61 CAD

### Shipping & Taxes Calculation
- **Highest shipping cost**: $6.16 CAD ($4.56 USD) for Pull-up Resistors
- **Canadian HST/GST (13% average)**: $34.01 CAD ($25.17 USD)

### Final Totals
- **Subtotal**: $193.59 USD / $261.61 CAD  
- **Taxes (13%)**: $25.17 USD / $34.01 CAD  
- **Shipping**: $4.56 USD / $6.16 CAD  
- **🔹 TOTAL**: **$223.32 USD / $301.78 CAD**