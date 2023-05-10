# import necessary modules
import tensorflow as tf
import pandas as pd
import tkinter as tk

#new object of tk class
main_window = tk.Tk()

# title of our window
main_window.title("ConcreForce")

# size, color of the window
main_window.geometry("1200x800")
main_window.config(bg = "#000000")

# sample label
label = tk.Label(main_window, text = "ConcreForce - AI Concrete Strength Predictor", font=("Arial", 22))
label.pack()

# label for cement
label_cement = tk.Label(main_window, text="Cement (kg/m^3):", font=("Arial", 14))
label_cement.place(x=100, y=100)

# input for cement
cement_input = tk.DoubleVar()

# entry widget for the input
cement_entry = tk.Entry(main_window, textvariable=cement_input, font=("Arial", 14))
cement_entry.place(x=400, y=100)

# label for slag
label_slag = tk.Label(main_window, text="Slag (kg/m^3):", font=("Arial", 14))
label_slag.place(x=100, y=150)

# input for slag
slag_input = tk.DoubleVar()
# entry widget
slag_entry = tk.Entry(main_window, textvariable=slag_input, font=("Arial", 14))
slag_entry.place(x= 400, y= 150)

# label for fly ash
label_fly_ash = tk.Label(main_window, text="Fly ash (kg/m^3):", font=("Arial", 14))
label_fly_ash.place(x=100, y=200)

# input for flyash
fly_ash_input = tk.DoubleVar()
# entry widget
fly_ash_entry = tk.Entry(main_window, textvariable=fly_ash_input, font=("Arial", 14))
fly_ash_entry.place(x= 400, y= 200)

# label for water
label_water = tk.Label(main_window, text="Water (kg/m^3):", font=("Arial", 14))
label_water.place(x=100, y=250)

# input for water
water_input = tk.DoubleVar()
# entry widget
water_entry = tk.Entry(main_window, textvariable=water_input, font=("Arial", 14))
water_entry.place(x= 400, y= 250)

# label for superplasticizer
label_superplasticizer = tk.Label(main_window, text="Superplasticizer (kg/m^3):", font=("Arial", 14))
label_superplasticizer.place(x=100, y=300)

# input for superplasticizer
superplasticizer_input = tk.DoubleVar()
# entry widget
superplasticizer_entry = tk.Entry(main_window, textvariable=superplasticizer_input, font=("Arial", 14))
superplasticizer_entry.place(x= 400, y= 300)

# label for coarse aggregate
label_coarse_aggregate = tk.Label(main_window, text="Coarse Aggregate (kg/m^3):", font=("Arial", 14))
label_coarse_aggregate.place(x=100, y=350)

# input for coarse_aggregate
coarse_aggregate_input = tk.DoubleVar()
# entry widget
coarse_aggregate_entry = tk.Entry(main_window, textvariable=coarse_aggregate_input, font=("Arial", 14))
coarse_aggregate_entry.place(x= 400, y= 350)


# Input label for fine aggregate
label_fine_aggregate = tk.Label(main_window, text="Fine Aggregate (kg/m^3):", font=("Arial", 14))
label_fine_aggregate.place(x=100, y=400)

# input for fine_aggregate
fine_aggregate_input = tk.DoubleVar()
# entry widget
fine_aggregate_entry = tk.Entry(main_window, textvariable=fine_aggregate_input, font=("Arial", 14))
fine_aggregate_entry.place(x= 400, y= 400)

# Input label for age
label_age = tk.Label(main_window, text="Age (days):", font=("Arial", 14))
label_age.place(x=100, y=450)

# input for age
age_input = tk.DoubleVar()
# entry widget
age_entry = tk.Entry(main_window, textvariable=age_input, font=("Arial", 14))
age_entry.place(x= 400, y= 450)

# load the data
data = pd.read_csv('Concrete_Data.csv')

# splitting the data into input and output variables
x = data.drop("csMPa", axis = 1)
y = data["csMPa"]




from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.80, shuffle=True, random_state=42)

# preprocessing (normalizing the data)
# we'll see if we need this later
# (we dont, normalizing it made it worse by a mae of 2)

#making the model

model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation = "relu"),
    tf.keras.layers.Dense(40, activation = "relu"),
    tf.keras.layers.Dense(30, activation = "relu"),
    tf.keras.layers.Dense(15, activation = "relu"),
    tf.keras.layers.Dense(10, activation = "relu"),
    tf.keras.layers.Dense(5, activation = "relu"),
    tf.keras.layers.Dense(1,  activation = tf.keras.activations.linear)
])

# compile the model

model.compile(loss=tf.keras.losses.mean_absolute_error,
                    optimizer=tf.keras.optimizers.Adam(learning_rate= 0.001),
                    metrics=["mae"])

# fit the model

#history = model.fit(x_train, y_train, epochs=300, verbose=1, validation_data=(x_test, y_test) )

#model.save('concrete_strength_prediction')

model = tf.keras.models.load_model('concrete_strength_prediction.h5')


output_label = tk.Label(main_window)
output_label.place(x = 200, y = 600)
output_label.config(width=50, height=2)



def display ():
    user_data = pd.DataFrame({'cement': cement_input.get(),
                              'slag': slag_input.get(),
                              'flyash': fly_ash_input.get(),
                              'water': water_input.get(),
                              'superplasticizer': superplasticizer_input.get(),
                              'coarseaggregate': coarse_aggregate_input.get(),
                              'fineaggregate': fine_aggregate_input.get(),
                              'age': age_input.get()},
                             index=['0'])
    output_label.configure(text="The estimation is " + "{:.3f}".format(float(model.predict(user_data))) + " MegaPascal", font= ("Arial", 15))

estimate_button = tk.Button(main_window, text="Estimate!",font="Arial", command=display, width= 10, height= 5)
estimate_button.place(x = 900, y = 300 )
main_window.mainloop()


