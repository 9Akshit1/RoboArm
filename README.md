# RoboArm
DESCRIPTION: RoboArm is just like how it sounds. It is a robotic arm with 3 joints that can mimic real arm movements. It uses servo motors controlled by an EMG sensor to move based on brain signals. The arm can also be controlled manually via a computer or preset motions for easy testing and flexibility.

INSPIRATION: I've always been inspired by how human arms move and wanted to recreate that motion using motors and tendons. Additionally, I previosuly built a RoboHand so to add on to that project, I wanted to add a RoboArm to extend it. I also wanted to get into the neuroscience field, so this is my first step into it!

# Final CAD
The separate part files and body files are in ther CAD folder. The full built hand CAD File is called FULL_Robotic_Arm.step in the CAD folder. 

Also, PLEASE NOTE that I can always just add supports before I 3D print, so we do not need to worry about any floating or thin pieces because they were likely intended by me to be there!

Additionally, I made two versions of my arm which are the standard robotic arm verison (the type of design you would see when you search up a robotic arm), and the cooler human-like arm version (which was designed after my arm). The human arm version is a little more messy and likely contains a few errors that I need to fix, which is why I hope the reviewer (or whoever is reading this) can ignore the mistakes in the human arm version. I will first print out the robotic arm version anyways, and only AFTER I fix the human arm version and test it so that it works in the CAD software, will I print it. 
TLDR: Please ONLY look at the Robotic Arm version, because I will work on that version first, and then after fixing up the human arm version, I may or may not print it.

The FULL Robotic Arm and Human Arm.stl file also contains all of the electronics components that would be ON the robot, which are literally only the servo motors, because for everything else, I will likely keep it wired separately, and just run one set of clean wires through the arm to the servo motors. 

The standard robotic arm version:

![Robotic Arm Base](https://hc-cdn.hel1.your-objectstorage.com/s/v3/f9fd9571f1704dda48574c137284a2ef454db6de_robotic_arm_base_cad.png)

![Robotic Arm First Link Half 1](https://hc-cdn.hel1.your-objectstorage.com/s/v3/5eae04ccda41d3b6a81594be1c96335aac3a6bb0_robotic_arm_first_link_half_1_cad.png)

![Robotic Arm First Link Half 2](https://hc-cdn.hel1.your-objectstorage.com/s/v3/7d3e892d04aba3d37122b870674cba6414d20ec9_robotic_arm_first_link_half_2_cad.png)

![Robotic Arm Second Link Half 1](https://hc-cdn.hel1.your-objectstorage.com/s/v3/df17b024f6a9cc19f233cf4dafafec793ab8b46a_robotic_arm_second_link_half_1_cad.png)

![Robotic Arm Second Link Half 2](https://hc-cdn.hel1.your-objectstorage.com/s/v3/6c9c4639559926d0e712b3b1133b889ff4c34145_robotic_arm_second_link_half_2_cad.png)

The cool human arm version:

![Human Arm Base](https://hc-cdn.hel1.your-objectstorage.com/s/v3/e1a96042fae8d9c8134b4bd1a5ce0a0e1372ae05_human_arm_base_cad.png)

![Human Arm Base Last Part](https://hc-cdn.hel1.your-objectstorage.com/s/v3/c043a35d415e0ba23a2177244c7650fd0d468701_human_arm_base_last_part_cad.png)

![Human Arm First Link Half 1](https://hc-cdn.hel1.your-objectstorage.com/s/v3/81e493188060fc20efd78611f3549faeaf62b808_human_arm_first_link_half_1_cad.png)

![Human Arm First Link Half 2](https://hc-cdn.hel1.your-objectstorage.com/s/v3/84caaa68840309290025c52d0e979977475830c4_human_arm_first_link_half_2_cad.png)

![Human Arm Second Link Half 1](https://hc-cdn.hel1.your-objectstorage.com/s/v3/1af024615059c17485921f87988af8b42200e54a_human_arm_second_link_half_1_cad.png)

![Human Arm Second Link Half 2](https://hc-cdn.hel1.your-objectstorage.com/s/v3/1ab4e53e4b3da52219abc0f4ab6d51cbe22329c4_human_arm_second_link_half_2_cad.png)

The wrist is the same for both versions:

![Wrist CAD](https://hc-cdn.hel1.your-objectstorage.com/s/v3/e4a2a2b5b410058aa683b52b27b036a42f505c5d_wrist_cad.png)

This is what the full build hand will look like. I will likely hot glue every piece together. 

The standard robotic arm version:

![FULL CAD Robotic Arm](https://hc-cdn.hel1.your-objectstorage.com/s/v3/59bf4d3f97b163ade5a52861d0b129dd9797eac7_full_robotic_arm_cad.png)

The cool human arm version:

![FULL CAD Human Arm](https://hc-cdn.hel1.your-objectstorage.com/s/v3/7f08667ebc01651b58fe639efda7e46ed0e29a6a_full_human_arm_cad.png)

# Final Circuit Schematic
Schematic is called RoboArm.kicad_sch in the schematic_pcb/RoboArm folder.
The schematic shows that I used 10 EMG sensors, however in reality, I will likely buy only 6 in the beginning. However, nothing much changes in the schematic except for deleting the 4 extra EMG sensors, so please excuse that. In the future, when I do buy the other 4 sensors out of pocket, I want to be able to use the 10 EMG sensor schematic easily, which is why I have not changed it to have 6 EMG sensors.
Additionally, for the LM2596S-ADJ DC Buck Converter, the OUT pin has a No Connect flag because I was having an error before with the OUT pin connected to the PWR_FLAG, and it didn't seem fixable even with the help of others. So, instead, I just disconnected the OUT pin and then wires the PWR_FLAG to a +5V power symbol instead, as the OUT pin would function as a 5V power source anyways. In the real world, I'll connect the OUT pin, so we do not need to worry about that. Finally, the other two pins, ON/OFF and FB, have No Connect flags because the type of DC buck converter I am buying often has those things internally wired anyways, so there is no need to do additioanlly wirings for them in my case.

![Final Schematic](https://hc-cdn.hel1.your-objectstorage.com/s/v3/0d52475c7330d083d6516c378238f20f8c8fdde7_schematic_jy30.png)

# I am not using a PCB, because I will manually solder everything.

# Final Firmware Stuff
The firmware software is in the firmware folder. Theres not much of an output that I can show without the electronic parts, however, I have generated a simulated image dataset which will be used along with the visual camera and the IMUs. I also built a basic real time data capture & modality alignment, and a LSTM & RL model code, but they can't be run since I do not have the parts nor the real dataset.

Generated Simulated Dataset:

![Scene 0 in Image Dataset](https://hc-cdn.hel1.your-objectstorage.com/s/v3/aa28b4ffe4847cc6dde3fffceca2d866c1e15a62_scene_0000_center.png)

![Scene 1 in Image Dataset](https://hc-cdn.hel1.your-objectstorage.com/s/v3/64b79e58480662895451bf9c59326c34985e8c6b_scene_0001_center.png)

![Scene 2 in Image Dataset](https://hc-cdn.hel1.your-objectstorage.com/s/v3/ca07a6d5c91f754adc8886db1bd9013df21d6f7c_scene_0002_center.png)

## Bill of Materials (BOM)

| Component | Model/Part Number | Qty | Price (CAD) | Price (USD) | Link | Notes |
|-----------|-------------------|-----|-------------|-------------|------|-------|
| ESP32 DevKit V1 | ESP32-WROOM-32 | 1 | $7.11 | $5.22 | [Link](https://www.aliexpress.com/item/1005008503831020.html?spm=a2g0o.productlist.main.1.20dd1387tCczqQ&algo_pvid=e6ab4063-bcd5-4c34-b912-96113803fa55&algo_exp_id=e6ab4063-bcd5-4c34-b912-96113803fa55-0&pdp_ext_f=%7B%22order%22%3A%22381%22%2C%22eval%22%3A%221%22%7D&pdp_npi=4%40dis%21CAD%216.63%211.63%21%21%2133.72%218.29%21%402101ead817540086983356692e0fa4%2112000045457574256%21sea%21CA%216438900822%21ABX&curPageLogUid=RJRDaCKEh084&utparam-url=scene%3Asearch%7Cquery_from%3A) | Main controller |
| ESP32-CAM | ESP32-CAM | 1 | $12.33 | $9.05 | [Link](https://www.aliexpress.com/item/1005006341099716.html?spm=a2g0o.productlist.main.4.10b31f44f3CRYE&algo_pvid=cf5cc23f-74d1-44a8-bcb2-311bc13f1dcc&algo_exp_id=cf5cc23f-74d1-44a8-bcb2-311bc13f1dcc-3&pdp_ext_f=%7B%22order%22%3A%22103%22%2C%22eval%22%3A%221%22%7D&pdp_npi=4%40dis%21CAD%212.83%211.63%21%21%2114.42%218.32%21%402101d9ee17540044723562302eab01%2112000036821212387%21sea%21CA%216438900822%21ABX&curPageLogUid=s9SbCAg5Mvaq&utparam-url=scene%3Asearch%7Cquery_from%3A) | Computer vision |
| MG90S Micro Servos | TowerPro MG90S | 2 | $5.70 | $4.18 | [Link](https://www.aliexpress.com/item/1005008626768357.html?spm=a2g0o.productlist.main.2.7f2015caxNFEhm&aem_p4p_detail=202507311630421114710384897480003208335&algo_pvid=f808fc16-865d-4975-95a1-d6fab4f69bc5&algo_exp_id=f808fc16-865d-4975-95a1-d6fab4f69bc5-1&pdp_ext_f=%7B%22order%22%3A%22317%22%2C%22eval%22%3A%221%22%7D&pdp_npi=4%40dis%21CAD%215.83%211.63%21%21%214.12%211.15%21%402101c59517540046428904607ef048%2112000046009069094%21sea%21CA%216438900822%21ABX&curPageLogUid=JxTQbST0aPKs&utparam-url=scene%3Asearch%7Cquery_from%3A&search_p4p_id=202507311630421114710384897480003208335_1) | Cheapest price servo motor. These motors are for the wrist |
| LX-225 Servo Motors | LX-225 | 4 | $94.82 | $69.60 | [Link](https://www.alibaba.com/product-detail/Double-Shaft-LX-225-High-Speed_1600996726890.html?spm=a2700.galleryofferlist.normal_offer.d_title.318813a0PIdYL7&selectedCarrierCode=SEMI_MANAGED_STANDARD@@STANDARD) | 25 kg.cm torque servos. Originally wanted Dynamixel XM430 servos, but they were too expensive, so this is the best servo motor that has a decent price and good quality and usability for my application. Unit price is $17.41 USD |
| PWM Driver | Adafruit PCA9685 | 1 | $4.43 | $3.25 | [Link](https://www.aliexpress.com/item/1005006298833960.html?spm=a2g0o.productlist.main.2.787a5d9datmysx&aem_p4p_detail=202507311606341701772608146720003198777&algo_pvid=15c1d215-a59f-48b5-8adb-4727f7aa4df9&algo_exp_id=15c1d215-a59f-48b5-8adb-4727f7aa4df9-1&pdp_ext_f=%7B%22order%22%3A%22186%22%2C%22eval%22%3A%221%22%7D&pdp_npi=4%40dis%21CAD%214.43%211.63%21%21%2122.51%218.27%21%402103247917540031941675191ebb89%2112000037529842018%21sea%21CA%216438900822%21ABX&curPageLogUid=XfzBycOLYrkp&utparam-url=scene%3Asearch%7Cquery_from%3A&search_p4p_id=202507311606341701772608146720003198777_1) | 16-channel PWM |
| Level Shifter | TXS0108E | 1 | $1.54 | $1.13 | [Link](https://www.aliexpress.com/item/1005006825517960.html?spm=a2g0o.productlist.main.6.6bb7552bbSnz1J&aem_p4p_detail=202507311733325343759419334840003304366&algo_pvid=597128eb-8e28-40bf-b7db-30e1878039f4&algo_exp_id=597128eb-8e28-40bf-b7db-30e1878039f4-5&pdp_ext_f=%7B%22order%22%3A%22113%22%2C%22eval%22%3A%221%22%7D&pdp_npi=4%40dis%21CAD%216.55%211.63%21%21%2133.32%218.30%21%402103244817540084122198878e2fa5%2112000046140260695%21sea%21CA%216438900822%21ABX&curPageLogUid=MUMJA940BGSI&utparam-url=scene%3Asearch%7Cquery_from%3A&search_p4p_id=202507311733325343759419334840003304366_2) | 3.3V to 5V converter. This is the best deal. |
| EMG Sensors | SEN-13723 RoHS MyoWare | 6 | $245.23 | $180.00 | [Link](https://www.alibaba.com/product-detail/SpotMyoWare-Muscle-Sensor-SEN-13723-Muscles_1601406726720.html?spm=a2700.galleryofferlist.normal_offer.d_title.52e313a0BHVrsO) | Muscle activity sensors. These are one of the best quality sensors suitable for research projects such as this and are also the cheapest deal I found everywhere! I need 6 because I'm doing a full hand-arm, meaning there are multiple muscles to look at and analyze and align and eveyrthing. Its essentially one for each DOF of the arm, however it should be ideally 1 or 2 more than this for proper accurate research and to account for the hands' muscles. However, that is outside of our budget which is why I'm only buying 6. |
| ADC Modules | ADS1115 | 3 | $5.54 | $4.07 | [Link](https://www.aliexpress.com/item/1005007628692389.html?spm=a2g0o.productlist.main.2.6c913e84NYJmIs&aem_p4p_detail=202507311619095829272559576480003213401&algo_pvid=1275a360-73ba-4ddf-a858-0478d2ddf380&algo_exp_id=1275a360-73ba-4ddf-a858-0478d2ddf380-1&pdp_ext_f=%7B%22order%22%3A%22612%22%2C%22eval%22%3A%221%22%7D&pdp_npi=4%40dis%21CAD%2111.18%216.23%21%21%2156.85%2131.67%21%40210313e917540039492598181eaff1%2112000041563143884%21sea%21CA%216438900822%21ABX&curPageLogUid=H8gnUD0dSVdM&utparam-url=scene%3Asearch%7Cquery_from%3A&search_p4p_id=202507311619095829272559576480003213401_1) | 16-bit ADC |
| IMU Sensor | BNO055 | 1 | $11.43 | $8.39 | [Link](https://www.aliexpress.com/item/1005005506735089.html?spm=a2g0o.productlist.main.1.52fe741baPvnr7&algo_pvid=729d77a6-79cf-4867-bc7c-fe69d8e9dd75&algo_exp_id=729d77a6-79cf-4867-bc7c-fe69d8e9dd75-0&pdp_ext_f=%7B%22order%22%3A%22398%22%2C%22eval%22%3A%221%22%7D&pdp_npi=4%40dis%21CAD%2111.43%215.77%21%21%218.08%214.08%21%402101ef5e17540042270443170e80cc%2112000033345589764%21sea%21CA%216438900822%21ABX&curPageLogUid=OxnT7GIlahPs&utparam-url=scene%3Asearch%7Cquery_from%3A) | 9-DOF orientation |
| Power Supply | 7.4V 20A SMPS | 1 | $19.99 | $14.67 | [Link](https://www.amazon.ca/AMZZN-2000mAh-Battery-Charging-Engineering/dp/B0D1G8L8Y4/ref=sr_1_3?crid=2KCC7GMAS4ZGX&dib=eyJ2IjoiMSJ9.164idcALTh0_bZAf5Vi-Q6R-2Q0STTdt4w2c81RtTbtaUKnLO4Ucd45u6AvaqkE0OeTMT9IspAq8Ri0kZ8s2mcJOSywLseU-L-a-Z7pWwmSMrADnVMfCIFGfrfOMK2zntBF9364_MNciDtu7PFy6ZrQAMe9qxGQYBjBZ5hpoOs65n0niSHpINcpWLVqBpRno.LJwbNIHeS8PQbgy97yjwZOc-AsPsnFNpiDhrpLOiaY4&dib_tag=se&keywords=7.4V%2B5A%2B2000mAh%2BSMPS%2Bbattery%2Bsupply&qid=1753988267&sprefix=7%2B4v%2B5a%2B2000mah%2Bsmps%2Bbattery%2Bsupply%2Caps%2C86&sr=8-3&th=1) | High current supply, and is best & cheapest deal. This Amazon deal was even better than the ones on Ali-express surprisingly, because the Aliexpress ones had really high shipping fees or really high prices compared to the Amazon one. |
| DC Buck Converter | LM2596S-ADJ | 1 | $0.63 | $0.46 | [Link](https://www.alibaba.com/product-detail/Dc-Dc-Converter-Lm2596s-Lm-2596_1600103532696.html?spm=a2700.galleryofferlist.normal_offer.d_title.50b813a0tZz484) | Adjustable. Handles high voltages and can convert to the 5V that I need it to be able to. Free shipping too. Best & cheapest deal.  |
| Pull-up Resistors | 4.7kΩ 1/4W | 2 | $2.41 | $1.77 | [Link](https://www.aliexpress.com/item/32317131954.html?spm=a2g0o.productlist.main.1.37bb59c9bNACJh&algo_pvid=a11e4016-b8e3-4bb5-afd2-0618ed42052f&algo_exp_id=a11e4016-b8e3-4bb5-afd2-0618ed42052f-0&pdp_ext_f=%7B%22order%22%3A%22173%22%2C%22eval%22%3A%221%22%7D&pdp_npi=4%40dis%21CAD%212.41%211.63%21%21%211.70%211.15%21%402101c5b217540055874202707e243c%2158290968831%21sea%21CA%216438900822%21ABX&curPageLogUid=QT6Bm5YEg1FW&utparam-url=scene%3Asearch%7Cquery_from%3A) | I²C pull-ups. Cheapest deal. The link has 10 pieces, but there was no product that sold them individually |
| Capacitor 1000µF | Electrolytic 25V | 1 | $4.37 | $3.21 | [Link](https://www.aliexpress.com/item/33010665515.html?spm=a2g0o.productlist.main.19.5696e336YqdJ1M&algo_pvid=05eaeade-4550-4688-b75d-17b6cb5b207e&algo_exp_id=05eaeade-4550-4688-b75d-17b6cb5b207e-18&pdp_ext_f=%7B%22order%22%3A%2270%22%2C%22eval%22%3A%221%22%7D&pdp_npi=4%40dis%21CAD%214.37%211.63%21%21%213.09%211.15%21%402101ef5e17540057514022423e8084%2167121829359%21sea%21CA%216438900822%21ABX&curPageLogUid=NYzuwQWBHlT4&utparam-url=scene%3Asearch%7Cquery_from%3A) | Power filtering. The link has 10 pcs but there was no other product that only sold 1 piece and this was the cheapest |
| Capacitor 470µF | Electrolytic 25V | 1 | $3.54 | $2.60 | [Link](https://www.aliexpress.com/item/1005009356203410.html?spm=a2g0o.productlist.main.7.1ba21966B6Bsfu&algo_pvid=d8d651a6-319c-4b37-9832-ad4487a8e9c2&algo_exp_id=d8d651a6-319c-4b37-9832-ad4487a8e9c2-6&pdp_ext_f=%7B%22order%22%3A%224%22%2C%22eval%22%3A%221%22%7D&pdp_npi=4%40dis%21CAD%213.54%213.54%21%21%2118.01%2118.01%21%402101e07217540060250496678e392f%2112000048852082722%21sea%21CA%216438900822%21ABX&curPageLogUid=QfCIRv5ez5KU&utparam-url=scene%3Asearch%7Cquery_from%3A) | Power filtering. Also, the link has 10 pieces, but there were no others that sold less than that and for this cheap price |
| Capacitor 220µF | Electrolytic 25V | 1 | $2.89 | $2.12 | [Link](https://www.aliexpress.com/item/1005006873397126.html?spm=a2g0o.productlist.main.4.5fa913bftkgyHG&algo_pvid=ec324fd1-1c9b-4682-9dc7-e1882c08c574&algo_exp_id=ec324fd1-1c9b-4682-9dc7-e1882c08c574-3&pdp_ext_f=%7B%22order%22%3A%2224%22%2C%22eval%22%3A%221%22%7D&pdp_npi=4%40dis%21CAD%212.89%212.89%21%21%2114.70%2114.70%21%402103247017540061461082516ee38c%2112000038584300721%21sea%21CA%216438900822%21ABX&curPageLogUid=3lv9bcZ5MJZZ&utparam-url=scene%3Asearch%7Cquery_from%3A) | Power filtering. Also, the link has 10 pieces, but there were no others that sold less than that and for this cheap price |
| Jumper Wires | Dupont Wires | A lot | N/A | N/A | N/A | I already have a lot of these |
| Breadboard | Half-size | 1 | N/A | N/A | N/A | I already have 2 empty ones to use |

---

## Cost Summary
### Component Totals
- **Subtotal**: $309.59 USD / $422.17 CAD

### Shipping & Taxes Calculation
- **Shipping cost**: 
        - From Amazon: $0.00 USD / $0.00 CAD --- Only product was the Battery and it has FREE SHIPPING
        - From Alibaba/AliExpress: $0.00 USD / $0.00 CAD  --- Every product is FREE SHIPPING because this is my first ever order on Alibaba/Aliexpress
        Total Shipping: $0.00 USD / $0.00 CAD
- **Canadian HST/GST (13%)**: $40.25 USD / $54.88 CAD

### Final Totals
- **Subtotal**: $309.59 USD / $422.17 CAD
- **Shipping Total**: $0.00 USD / $0.00 CAD
- **Taxes (13%)**: $40.25 USD / $54.88 CAD
- **🔹 TOTAL**: $349.84 USD / $477.05  CAD

Since the BOM is slightly below $350 USD, then in the case that, when I buy my parts, that the total price is slightly above $350, I will pay that extra bit.
