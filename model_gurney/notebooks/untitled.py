import cv2
import numpy as np
import csv

image_path = "image.png"
img = cv2.imread(image_path)
img = cv2.resize(img, (700, 700))

ref_points = []  
real_coords = []  
measured_points = []  
scales = None  

def save_to_csv(filename, data):
    """ Sauvegarde les points mesurés dans un fichier CSV """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["X réel", "Y réel"])
        writer.writerows(data)

def click_event(event, x, y, flags, param):
    global ref_points, real_coords, measured_points, scales, img

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(ref_points) < 2:
            ref_points.append((x, y))
            print(f"Point de référence {len(ref_points)} sélectionné : pixel ({x}, {y})")

            real_x = float(input(f"Valeur réelle en X du point {len(ref_points)} : "))
            real_y = float(input(f"Valeur réelle en Y du point {len(ref_points)} : "))
            real_coords.append((real_x, real_y))

            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  
            cv2.imshow("Graphique", img)

            if len(ref_points) == 2:
                (px1, py1), (px2, py2) = ref_points
                (real_x1, real_y1), (real_x2, real_y2) = real_coords

                scale_x = (real_x2 - real_x1) / (px2 - px1)
                scale_y = (real_y2 - real_y1) / (py2 - py1)

                x0 = real_x1 - px1 * scale_x
                y0 = real_y1 - py1 * scale_y

                scales = {"x": scale_x, "y": scale_y, "x0": x0, "y0": y0}
                print("\nÉchelles calculées. Cliquez maintenant sur les points à mesurer.")

        elif scales:
            real_x = scales["x0"] + x * scales["x"]
            real_y = scales["y0"] + y * scales["y"]
            measured_points.append((real_x, real_y))

            cv2.circle(img, (x, y), 5, (255, 0, 0), -1)  
            print(f"Point mesuré : pixel ({x}, {y}) → réel ({real_x:.2f}, {real_y:.2f})")
            cv2.imshow("Graphique", img)

cv2.imshow("Graphique", img)
cv2.setMouseCallback("Graphique", click_event)

print("\nAppuyez sur 'q' pour terminer la sélection des points.")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 'q' ou 'ESC'
        print("Fin de la sélection.")
        break

cv2.destroyAllWindows()
save_to_csv("points_mesures.csv", measured_points)