---
title: "RoboArm"
author: "Akshit Erukulla"
description: "A robotic arm that has 2–3 hinged segments moved by servos or tendons, controlled by a microcontroller and EMG sensor for mind-control."
created_at: "2025-07-18"
---
**Total Time spent: 2h**

# July 18th:
I worked on my CAD. I had to really think about how to design the arm. I decided to start off with some cylidners and then processeded to hollow them out, cut them, connect them, then added servo motor holders I think thsi is a good general structure, but I'll have to measure the mount position on a servo motor to exactly determine the positiosn I would put it in the real world. I also need to check how much the servo motors would be abe\le to rotate at the crrent positions.

![CAD Arm](https://hc-cdn.hel1.your-objectstorage.com/s/v3/2e864bb8f16b1cca8659abd531553ba01220fb8a_cad_jy_18.png)

**Time spent: 2h**

# July 21st:
I worked on my CAD. First of all, I copy-pasted the RoboHand I made in a previous project just to see how it would look. Then, I realized the hand was too big, so I measured my own hand's fingers and dimensions in order to apply that to the CAD. I also moved everything closer so it fit together. I realized I had to add covering for every single part, so I did that, and had to adjust the dimensions of the white cylinders of the arm because I realised most of them were too big. It was difficult trying to make space for the servo motors, and probably will be even more difficult, because I realised I need to add more servo motors to the hand itself, since it needs to be able to move all the fingers very freely. 
I also realzied I need to make the right ends of the cylinders circular and then make their left ends have a circular cutout, so the joints can move nicely.

![CAD Arm](https://hc-cdn.hel1.your-objectstorage.com/s/v3/93613baa2f7cda7798a47aaf6605544a49b3bef3_image.png)

**Time spent: 1.5h**

# July 22nd:
I worked on my CAD. I first made a literal human arm within the CAD which was done through using scanned arm models and then converting them to STL files and then hollowing them out while also removing the extra stuff around them. After that, I decided to work on modeling exactly what the servo motors and inner components of my RoboARM would look like, because my previous version (the arm in the middle) felt too inefficient and also didn't satisfy everything I wanted to. Instead, one of the major improvements I did was build a 3-axis gimbal at the base of the arm which I made following a tutorial. After researching I decided to use Dynamixel motors for the gimbal and the elbow. Then, at the wrist, I had to also change the motor to a standard MG90S. These motor changes required to change the casing that would contain the motors. Anyways, for the wrist, I had to also make a curved cutout the blue wrist/palm part because I accounted for the hand's actual yaw movement.
Everything in total took so much time because I completed multiple tasks, which all required detail, like the arm, the gimbal, and figuring out my exact components, getting their dimensions and their horns.

![CAD Arm](https://hc-cdn.hel1.your-objectstorage.com/s/v3/97acb80c390be0719a95e994a2181757fff5a221_cad_jy22.png)

**Time spent: 6h** (I'm not kidding, it actually took that long)

# July 23rd:
I worked on my CAD again. While the image looks messy, all it really is that I merged the "revised components" version of the arm (at the back) into the actual human arm model. I had a lot of trouble with the gimbal because the little stage in the middle was too small to contain the whole arm. Moreover, the motion of each of the servo motors in the gimbal have their own area/circle of movement so I had to estimate that, then make cylinders of them, to then cut out of the big red base box. I was also in general trying to figure out how the big arm would fit on the stage, and in the end, finalized on a tube + cone that connects the stage to the arm. Then, I moved on to the elbow which also surprisingly took a long time. I had a lot of trouble actually figuring out the proper position and attachment method for my servo motor to act as a hinge. This was because I really wanted the arm to be a fully enclosed and connected thing like how it is in the real world, but when you move the arm around the hinge, it would get "stopped" by the bicep part. I'm still not satisfied with what I decided to do (which was just leave a space at the hinge for the servo motor, so the arm could move a full 120 degrees which is what the orange arm is showing).
Moreover, if you look at the "component" arm at the back and look at the hand, I changed the hand quite a bit. I had to significantly shorten it and I also was trying to figure out how to give the thumb 3 DOF movement, but I realized that the servo motors wouldn't be able to fit. I'm not done with the hand, but technically, that was for another project, and the 3 DOF isn't completely necessary, so I don't need it right now, but I still spent time on it, because the thumb movement is very important for a proper human hand. Most robotic hands don't have a 3 DOF thumb because of these problems, but I will eventually fix that.
I was going to work for another hour, but my family forced me to take a break. :(

![CAD Arm](https://hc-cdn.hel1.your-objectstorage.com/s/v3/2a9f5c28595686b27b55a0872b97e1e1f2864aa3_cad_jy_23.png)

**Time spent: 5h** 

# July 24th:
I worked on my CAD again. I think I've finally finished. Basically, I realized that the gimbal was really messed up, and it still is. I tried moving the bicep part around each of the gimbal's servo motors and I realized that the bicep was going to be stopped by the servo motors, especially the bottom (far right) motor because the arm is very big and wide. So I had to extend the connectors of the gimbal for almost each servo. I'm still not sure if it would work, but I do know that even if it doesn't completely fit the whole range of a human arm, it will still be able to complete a majority of it. Anyways, the wrist part was relatively simple, I just had to move the components to the human arm's wrist, then adjust the servo motor so it wouldn't waste too much space.
Then, I decided that I want an actual more simplistic and straightforward "robotic" arm version, instead of the realistic human arm version as well. So, I just duplicated the white cylinders of yesterday's middle arm (the original arm that I built which was bad), and then put it around the orange cylinders. I then had to fill it in, so they would actually connect, and also made sure to shorten the cylinders accordingly so they wouldn't interfere with the realistic movement amount of the arm.
I think tomorrow, I'll work on separating the parts and fixing anything else I notice.

![CAD Arm](https://hc-cdn.hel1.your-objectstorage.com/s/v3/0667d4082a4b8ca22801f61ec70032863b29ee40_cad_jy24.png)

**Time spent: 4h** 

# July 25th:
I worked on my circuit schematic. It was extremely confusing because some of the parts I neededcl ike the Dynamixel servo motors and the MyoWare sensors didnt have any footprints of  symbols, so I had to make them myself. Moreover, I had trouble getting the correct BNO055 and some of the symbosl like the ESP32-CAM had errors for their pins. They had Bidirectional as the electircal type for all of their pins, which is completely wrong because power pins like the 5V and the GND pins should be power input. Also, as I moved the components over everywhere, some of the wires got into unwanted junctions and stuff which was very very annoying to fix. There were also a lot of errors on the ERC so I had to track everything and fix it. There were so many components that I had to research how to connect and then manually connect everything as well. Since I'm a beginner at KiCAD, it took so much time to fix all of this.
And then I realized I had to add resisotrs forth e SDA and SCL lines, and capacitors for the motors, ADS1115s, and MyoWare sensors. 
After this, I decided to not use a PCB, because first of all, my components need to be spread out over the hand and arm, and also I will need to be careful to make sure the connections are actually correct IRL, so I'll manually solder everything.
I think the schematic is mostly done because I made sure all of the connections made sure by verifying what other youtube videos have done.

![Circuit Schematic](https://hc-cdn.hel1.your-objectstorage.com/s/v3/09242684435c8adfca3481af5a91649d5c3d4354_schematic_jy25.png)

**Time spent: 4h** 