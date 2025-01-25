#include <iostream>
#include <string>
#include <vector>

class Account {
protected:
    std::string accountNumber;
    std::string accountHolderName;
    double balance;

public:
    Account(const std::string &accNum, const std::string &holderName, double initialBalance)
        : accountNumber(accNum), accountHolderName(holderName), balance(initialBalance) {}

    virtual ~Account() {}

    virtual void deposit(double amount) {
        balance += amount;
        std::cout << "Deposited " << amount << " to account " << accountNumber << ". New balance: " << balance << std::endl;
    }

    virtual void withdraw(double amount) {
        if (amount > balance) {
            std::cout << "Insufficient balance in account " << accountNumber << std::endl;
        } else {
            balance -= amount;
            std::cout << "Withdrew " << amount << " from account " << accountNumber << ". New balance: " << balance << std::endl;
        }
    }

    virtual void display() const {
        std::cout << "Account Number: " << accountNumber << ", Account Holder: " << accountHolderName << ", Balance: " << balance << std::endl;
    }

    double getBalance() const {
        return balance;
    }
};

class SavingsAccount : public Account {
private:
    double interestRate;

public:
    SavingsAccount(const std::string &accNum, const std::string &holderName, double initialBalance, double rate)
        : Account(accNum, holderName, initialBalance), interestRate(rate) {}

    void addInterest() {
        double interest = balance * (interestRate / 100);
        deposit(interest);
        std::cout << "Interest added: " << interest << ". New balance: " << balance << std::endl;
    }

    void display() const override {
        std::cout << "Savings ";
        Account::display();
        std::cout << "Interest Rate: " << interestRate << "%" << std::endl;
    }
};

class CheckingAccount : public Account {
private:
    double overdraftLimit;

public:
    CheckingAccount(const std::string &accNum, const std::string &holderName, double initialBalance, double overdraft)
        : Account(accNum, holderName, initialBalance), overdraftLimit(overdraft) {}

    void withdraw(double amount) override {
        if (amount > balance + overdraftLimit) {
            std::cout << "Overdraft limit exceeded for account " << accountNumber << std::endl;
        } else {
            balance -= amount;
            std::cout << "Withdrew " << amount << " from account " << accountNumber << ". New balance: " << balance << std::endl;
        }
    }

    void display() const override {
        std::cout << "Checking ";
        Account::display();
        std::cout << "Overdraft Limit: " << overdraftLimit << std::endl;
    }
};

int main() {
    std::vector<Account*> accounts;

    accounts.push_back(new SavingsAccount("SA123", "John Doe", 1000.0, 5.0));
    accounts.push_back(new CheckingAccount("CA123", "Jane Smith", 500.0, 200.0));

    for (Account* account : accounts) {
        account->display();
        std::cout << std::endl;
    }

    accounts[0]->deposit(500);
    accounts[1]->withdraw(600);

    SavingsAccount* sa = dynamic_cast<SavingsAccount*>(accounts[0]);
    if (sa) {
        sa->addInterest();
    }

    std::cout << std::endl;
    for (Account* account : accounts) {
        account->display();
        std::cout << std::endl;
    }

    for (Account* account : accounts) {
        delete account;
    }

    return 0;
}
